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
|-- FIGURE_STYLE.md            how every figure is built, after Tufte and
|                              Doumont. Read before writing a figure
|-- environment.yml            Pinned environment (see section 4)
|-- environment.lock.yml       Full transitive solve, osx-arm64
|-- notebooks/
|   |-- 01_CompareUQ_CreateData.ipynb    generate/read data, compute metrics
|   |-- 02_CompareUQ_AnalyzeData.ipynb   fit 6 methods, score by W1/W2/KS
|   |-- 03_CompareUQ_PerformPLCA.ipynb   2,500 pLCAs, downstream results
|   `-- 04_CompareUQ_ReduceMetrics.ipynb which characteristics matter, and
|                                        which method to use (Stage 2f)
|-- src/
|   |-- components.py          moment-targeted component families (Stage 2a)
|   |-- mixture.py             the truncated-mixture parent (Stage 2a)
|   |-- genconfig.py           every generation parameter (Stage 2a)
|   |-- generator.py           parent -> dataset, plus the validity filter
|   |-- corpus.py              generate, write and read a named corpus
|   |-- categorysplit.py       resolve a category into specifiable products,
|   |                          on metadata only (Stage 2a-3)
|   |-- empirical.py           prepare the empirical EC3 datasets
|   |-- modality.py            Silverman critical bandwidth, and the VISIBLE
|   |                          mode count that the generator is tuned against
|   |-- customstats.py         weighted statistics, distances, bandwidths
|   |-- datageneration.py      legacy generation helpers, empirical cleaning
|   |-- families.py            the support (0, inf), the parametric families,
|   |                          the weighted KDE with a CDF, and the
|   |                          estimators (Stage 2b)
|   |-- fitting.py             the six PEWT fits and W1 scoring
|   |-- comparison.py          the paper's method comparison: held-out W1, the
|   |                          tail check, ranks and characteristic curves
|   |-- materialclass.py       structural / envelope / other, from the category
|   |                          NAME only, so the comparison can be read by what
|   |                          a material IS (Stage 2c)
|   |-- figstyle.py            FIGURE_STYLE.md in code: palette, rcParams,
|   |                          direct labeling, the grayscale check
|   |-- metricreduction.py     which characteristics carry signal (Stage 2f):
|   |                          the candidate set and its transforms, explicit
|   |                          missingness, the size confound, two model
|   |                          families ranked by permutation importance, the
|   |                          tautology guard, and the binned and LOWESS
|   |                          curves with bootstrap bands that replace the
|   |                          rolling averages
|   |-- weighting.py           does the weighting scheme matter, per dataset
|   |                          (Stage 2d): the location/shape split of the
|   |                          uniform-to-variable W1, the named relative
|   |                          measure, and A_IQR from the KL2 paper
|   |-- flip.py                what a given W1 COSTS (Stage 2d): the
|   |                          common-random-numbers pLCA, model-to-model
|   |                          distances, and the calibration curve
|   |-- mixedpolicy.py         let the method vary BY MATERIAL (Stage 2j): the
|   |                          one-number rule as a NEW KEY over the six
|   |                          already-fitted models, the paired gain against
|   |                          the best fixed policy, the unreachable
|   |                          per-material ceiling, and the group-composition
|   |                          split that says why the gain is the size it is
|   |-- metricset.py           which downstream metric the paper leads with
|   |                          (Stage 2g): does a metric RECOVER the truth
|   |                          rather than merely being stable, the argmax
|   |                          agreement, the corrected cap normalization, and
|   |                          the thin-far-tail stress test
|   |-- plca.py                the pLCA CONSTRUCTION (Stage 2e): common random
|   |                          numbers, the group-size and material-use-
|   |                          intensity sweep, the cluster bootstrap, NRMSE
|   |                          with an interval, and the run against the TRUE
|   |                          parents
|   |-- recovery.py            the evaluation target (Stage 2c): W1 against the
|   |                          known parent, cross-validation, the
|   |                          fit-versus-definitional split, regret,
|   |                          post-stratification, the paired bootstrap
|   |-- datavisualization.py   one color helper
|   |-- funcs_unit_conversion.py  EC3 unit normalization
|   `-- dct_metriclabels.json  display labels for the 22 metrics
|-- audits/                    one-off measurement scripts, each named for what
|                              it measures; see audits/README.md
|-- data/processed/            inputs, see section 5
|-- outputs/tables/            tidy results, see section 6
|-- outputs/figures/           publication and supplementary figures
|-- reports/                   handoffs, baselines, discrepancy log
`-- tests/                     regression, determinism, unit, notebook guards
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
37, 38 and 40 and `audits/lognormal_offset.py`.

`FAMILIES` also holds `lognormal_2p`, `lognormal_offset` and `gamma`, which are
reported alongside rather than used: `audits/family_comparison.py`.

### The bandwidth, and why W1 cannot choose it

`BW_METHOD = 'silverman_guarded'`, by decision 54. `customstats.weighted_bw`
also offers `'scott'`, which the study used through Stage 2a and which is now
reported as a sensitivity, and `'silverman'`, the rule the author's KL2 paper
uses.

**W1 falls monotonically as the KDE bandwidth shrinks**, to about 2 percent of
any standard rule, because a KDE with a vanishing bandwidth IS the empirical
distribution it is scored against. So W1 cannot choose a bandwidth and cannot
arbitrate between methods of different flexibility. Use leave-one-out
likelihood cross-validation for that; `audits/bandwidth_rules.py` has it.

**Silverman's rule breaks at SMALL n, not on small interquartile ranges.** Where
`(IQR/1.34)/sd` is smallest -- heavy-tailed categories with a tight core -- it
beats Scott on held-out likelihood every time, and flooring the robust scale
makes things worse. Where it fails is n = 3 to 10, because the quartiles are
interpolated between two order statistics. `'silverman_guarded'` is Silverman's
rule throughout, `0.9 * scale * n_eff ** -0.2`, with the SCALE ESTIMATE guarded:
the robust `min(sd, IQR/1.34)` at or above `SILVERMAN_MIN_NEFF = 20` effective
observations, the plain standard deviation below it. **It is NOT Scott below the
threshold**: the coefficient stays 0.9 throughout where Scott's is 1.06, and two
comments said otherwise until Stage 2c.

**The threshold was 30 until Stage 2c and is now 20** (decision 80, superseding
75). It was swept on BOTH the held-out likelihood it was chosen by and W1 against
the known parent, which disagree -- the first peaks at 20 to 30, the second at 5.
Stepping down one value at a time, every step from 200 to 20 is free or better
than free and the step 20 to 18 is the first that costs more than it buys.
`audits/guard_threshold_sweep.py`.

### The normality statistic is Shapiro-FRANCIA, under both weightings

`customstats.shapiro_francia_weighted`, and the columns are `fit_norm_SF` and
`fit_lognorm_SF`. Until Stage 2f the function was `shapiro_wilk_weighted` and
it returned scipy's true Shapiro-WILK W when the weights were uniform and a
Shapiro-Francia W' when they were not, so `fit_norm_SW` and `fit_norm_SW_uw`
were two DIFFERENT statistics that four panels of the characteristic figure
compared as though they were one.

**Shapiro-Francia is the only one of the two with a weighted form**, so it is
the only choice under which the uniform-versus-variable comparison is one
statistic under two weightings. Decision 125.

**The old claim that the two agree for n >= 20 is about right there and is not
the relevant range.** Median absolute difference: 0.010 at n = 4 to 10, 0.006
at n = 20, 0.0004 at n = 2,000. The smallest size stratum in this study is
n = 3 to 9.

`shapiro_wilk_scipy` is the true Shapiro-Wilk, kept for
`audits/shapiro_estimator.py` and called from nothing in the production path.
`_royston_pvalue` was wrong in two ways until Stage 2f and is now correct to
4e-12 against scipy; the statistic this module returns gets Royston's (1993)
Shapiro-FRANCIA transform instead, which declines to extrapolate outside
5 <= n_eff <= 5000. Nothing reported has ever used a p-value. Decision 126.

### Three scores per fit, and why one is not enough

`src/comparison.py`, called from notebook 2's final section, which is where the
method comparison the PAPER makes now lives. Audits under `audits/` stay what
they are: one-off measurements that get decided once.

| column | what it is for |
|---|---|
| `w1` | the study's criterion, in sample. Every reported number has always been this |
| `w1_heldout` | fitted on half the values, scored on the other half, both directions, several splits, **all six methods sharing each split so the comparison is paired**. In-sample W1 rewards flexibility and the families here run from 2 parameters to effectively n |
| `model_sd_ratio` | the fitted model's standard deviation over the data's. W1 is nearly blind to tail mass and the pLCA SAMPLES from these models; entry 43 is the failure this catches |

**Held-out W1 is for comparing FAMILIES, not for choosing a bandwidth.** It
removes most of W1's bandwidth sensitivity rather than replacing it with a sharp
optimum: measured on the empirical arm its median moves only from 0.233 to 0.243
across a 50-fold bandwidth range. Leave-one-out likelihood is the sharp
instrument for bandwidth and is what decision 54 used.

**The recovery score, Stage 2c.** `src/recovery.py` scores each fitted model
against the parent the dataset was drawn from, which is the cleanest test
available and needs no training data in the target. It exists on the SYNTHETIC
arm only, so it does not replace the in-sample score; the empirical arm's
equivalent is cross-validation. TWO comparisons come out of it and they answer
different questions: `w1_parent` scores each method against the parent IT is
estimating, which is the fair way to judge an estimation method, and `w1_market`
scores all six against the market-weighted parent, which is the only way to
compare the two WEIGHTING schemes, because only then are they estimating the
same thing.

### Fitting by the criterion we score by

`fit_family(name, x, w, method='w1')` minimizes W1 directly instead of the
likelihood, starting from the MLE fit so it can never score worse. Every
parametric family in this study is estimated by maximum likelihood and judged by
W1, which are different criteria, so a family could lose the comparison because
it was never fitted under the rule it is judged by. `FIT_METHOD = 'mle'` is the
study's method; the W1-optimal results are reported beside it, not instead.

**Scoring.** Every model is scored against the **variable-weighted** empirical
CDF, including the uniform-weighted fits. `score_grid_open` is
`SCORE_GRID_POINTS` points from `hi / npoints` to `hi = max(x) + 10 * spread`,
**open at zero**: the lower bound is the first point of the grid's own lattice,
chosen that way so that it is not a new free parameter.

**20,000 POINTS AND TRAPEZOID QUADRATURE, both settled in Stage 2c** (decision
81). At 1,000 points the criterion is not converged, and the error is
METHOD-DEPENDENT: it inflated the KDE's score by 3 to 5 percent against 0.2
percent for the lognormal, because the KDE's CDF has the most structure at grid
scale. `fitting.W1_ROUTE` selects how the integral is taken on that grid --
`'atoms'` discretizes the model's density into weighted points and takes a
discrete Wasserstein distance, `'trapezoid'` integrates `|F_model - F_data|`
directly. **The atom route never converges**, because adding points does not
extend the grid and a model with mass past its top keeps losing it: its p99
relative error sticks at 0.0379 from 20,000 points through 100,000 while
trapezoid reaches 0.0002. `audits/scoring_grid_error.py`.

**There is ONE implementation and notebooks must call it.** Notebook 2 cell 23
computed W1 inline for the synthetic arm and so silently kept the old quadrature
when `W1_ROUTE` moved, putting two different values for one quantity in two
tables. `tests/test_regression.py::test_synthetic_fits_and_w1_recomputed` caught
it and is the guard.

### What a W1 costs, and the randomness that hides it

`src/flip.py`, Stage 2d. Every score in this study is a distance; this is what a
distance DOES. The probability that a probabilistic LCA names a different
largest contributor crosses 1, 5 and 10 percent at relative W1 of **0.0029,
0.015 and 0.032**, in units of the dataset's own unweighted mean.
`flip.FLIP_THRESHOLDS` carries them.

**THOSE THREE VALUES WERE 0.0018, 0.011 and 0.025 UNTIL 2026-09-25 AND THIS
FILE PRINTED THE OLD ONES UNTIL STAGE 3.** They were calibrated on the corpus
that Stage 2h replaced, and on the regenerated corpus all three had fallen
outside their own recomputed 95 percent intervals, by factors of 1.62, 1.37 and
1.26. The recalibration is decision 223; notebook 1 reads the constant and was
re-run on it. They rose because the new corpus is more dispersed, so a given
flip probability corresponds to a larger absolute model distance -- the same
scale effect that raised every goodness-of-fit score without any fit getting
worse. `tests/test_flip.py::test_stored_flip_thresholds_sit_inside_their_own_intervals`
now fails if they drift again.

**HOW MANY DIGITS TO PRINT IS NOT A FIXED COUNT**, by decision 175: every
crossing is fitted twice, once with a logistic and once with an isotonic fit,
and prose rounds at the first figure the two disagree on. `flip.prose_crossing`
is that rule. Stage 3 added one guard to it -- the printed spread may not be
more than twice the real one -- after it rendered 0.015021 and 0.013899, which
agree to 7.5 percent, as "0.02 against 0.01". Both of decision 175's published
examples are unchanged.

**THE STUDY'S pLCA GIVES EACH METHOD ITS OWN RANDOM DRAWS.** Two methods are
therefore compared under two independent Monte Carlo samples. For every
CONTINUOUS output this is immaterial -- switching method changes a material's
estimated contribution by 0.18 where the average material contributes 1.00, far
more than resampling does. It matters only for an ARGMAX outcome such as "which
material has the highest rank-1 frequency", where two statistically tied
materials can swap places; that is why the flip calibration of Stage 2d gives
both methods the same uniform draws, through `families.rvs_from_uniform`.
Installing the same thing in the study's own pLCA is Stage 2e's and is a
refinement, not a repair.

### Does the weighting scheme matter, per dataset

`src/weighting.py`, Stage 2d. Three measures, and the third is not the one the
practitioner statement is built on.

The uniform-to-variable W1 splits into a LOCATION term, `abs(weighted mean -
unweighted mean)`, which is the exact lower bound W1 obeys, and a SHAPE
residual. It is **mostly location**: median share 0.725 empirical and 0.804
synthetic, and 0.96 where datasets have 3 to 9 values.

The relative measure is W1 over the dataset's own UNWEIGHTED mean, which is what
the study has always reported without saying so, since every dataset is divided
by that mean before anything else. All three candidate denominators are computed
with uniform weights, because one taken under the variable weights would move
with the quantity being measured and a practitioner cannot compute a
market-weighted mean without the market shares.

**A_IQR is KL2's measure and it is DOMINATED BY DATASET SIZE.** Across the
empirical arm it correlates with log size at -0.946 and with the coefficient of
variation at +0.042, and it moves by a factor of 12 from the smallest categories
to the largest against 1.07 to 1.87 across the dispersion range within a band.

**Scale invariance is NOT the reason, and an earlier version of this file said it
was.** A_IQR is exactly invariant under rescaling the data, but so is the
mean-relative separation used instead, so invariance cannot distinguish them.
What does is the denominator: A_IQR measures the density's uncertainty against
that curve's own height and width, so the spread cancels twice and only the
weight sampling noise survives; the separation is an x-axis distance over the
mean alone, so the spread-to-mean ratio survives, and that ratio is the
coefficient of variation. At fixed n over a 27-fold change in spread,
`A_IQR * sqrt(n)` stays within 2.21 to 2.38 and `separation / coeffvar` within
0.114 to 0.138.

A_IQR is reported for consistency with the published paper, unnormalized and over
1,000 draws, both read off that paper. The practitioner number is the separation,
which correlates with dispersion at +0.731 and with log size at -0.545 -- BOTH
matter, and dispersion dominates only within a size band, where it runs +0.83 to
+0.96.

## 2b. The pLCA construction

Stage 2e. Three properties of the probabilistic LCA that were fixed by
accident rather than by decision, and what each is now.

### Common random numbers

**The pLCA draws ONE uniform variate per material per Monte Carlo iteration and
pushes it through every method's inverse CDF.** Independent ACROSS MATERIALS
within an iteration, because materials in a building are not rank-correlated
and sharing a variate across them would invent a correlation this study does
not claim; identical ACROSS METHODS, so two identical models produce identical
results and the Monte Carlo floor under a comparison is exactly zero. This is
the practice Henriksson et al. (2015) and Heijungs (2021) recommend for
comparative probabilistic LCA and that Marsh et al. (in press) use.

Before Stage 2e each method drew its own stretch of one stream.
**It is a refinement and not a repair**: a uniform is a uniform, so each
method's own marginal sample is unchanged in distribution and only the PAIRING
changes. `TABLE_CRNComparison.csv` measures what it is worth, output by output,
against the unpaired draws it replaces and against the floor of running one
method twice.

The capped-reduction strategy is paired too. `plca.PassUniforms` caches one
variate vector per (material, redraw pass) for the group, so two methods
redrawing the same iteration on the same pass use the same variate. The passes
are indexed because within a pass the variate is fixed, so a draw that lands
above the cap again must be given a different one or the loop cannot terminate.

### Material use intensity is a share vector

Every dataset is normalized to an unweighted mean of 1.0, so a material's mean
contribution IS its use intensity and the intensity vector carries all of the
between-material variation. The study's own construction sets every intensity
to 1.0, which makes the materials exchangeable and a ranking as fragile as it
can be made.

A building total is arbitrary, so the object being swept is a point on the
simplex: `plca.mui_dirichlet` draws shares from a symmetric Dirichlet and
scales them to a mean of 1.0, and an infinite concentration returns the equal
vector EXACTLY rather than approaching it. `plca.mui_from_ratio` gives the
deterministic checkpoints 1:1, 2:1, 10:1 and 100:1.

**Report against the observable, never against the concentration.** A Dirichlet
concentration means nothing to a reader and what it implies about dominance
changes with the number of materials, which would confound the two sweeps.
`plca.contribution_profile` returns the ratio of the largest mean contribution
to the second largest and the leading material's share of the total, both
computable from a quantity take-off and both fixed before any distribution is
fitted, so neither can move with the method under test.

### The five statements, and the interventions

Stage 2e, after author review. A probabilistic LCA makes five kinds of
statement and the study reported one and a half of them: **magnitude** (the
building total as a distribution, including the chance of meeting a budget),
**attribution** (each material's contribution and share), **information**
(which material's uncertainty dominates), **action** (what an intervention
delivers and how likely it is to) and **comparison** (whether one design beats
another). `plca.building_statement`, `reduction_statement` and
`comparison_statement`, all scored against the true parent on the same variates.

**THE SPECIFICATION CAP IS ONE ABSOLUTE VALUE PER MATERIAL and the old form
could not measure what the strategy is for.** It was the 75th percentile of each
METHOD'S OWN draws, so the six were asked about six different interventions --
and because each was capped at its own 75th percentile, exactly 25 percent of
iterations were capped under every method by construction, which forces the
signal to zero. It is now `plca.specification_cap`, the 75th percentile of the
values a specifier holds, which is what a practitioner can compute and which
exists on the empirical arm where no parent does. **This moved every `capecc_`
column**, and it forced the `(1 - capecc)` divisor out with it, because that
divisor was exact only under the old cap.

**A capped draw is taken EXACTLY from the model conditioned on being below the
cap, by inverse CDF.** Redrawing until a value lands below the cap samples from
the same conditional distribution, so `ppf(u * F(cap))` is the same thing in one
step; decision 50 had already settled that sampling here is never by rejection,
and that loop was the last rejection sampler in the study. It also cannot fail,
which the loop could and did.

**There are four reduction strategies and only three need code.** Use less of a
material, specify a better product, substitute a different one -- and obtain a
supplier-specific declaration, which is exactly what the uncertainty index
measures and is the only one of the four that reduces the VARIANCE of the answer
rather than its level.

### The pLCA against the truth

`plca.truth_run` runs the same groups twice on the same variates: once with the
fitted models and once with the datasets' TRUE parents, recovered by replaying
the generator. The difference is the error the fitted model causes and contains
no Monte Carlo noise at all.

The truth is the MARKET-weighted parent, because a probabilistic LCA of what
gets built is a statement about the population weighted by production, and it
is the one population all six methods can be scored against on equal terms. The
sampling parent is reported beside it, which is what a uniform-weighted method
is estimating, so the definitional part of each method's error is visible.

`plca.LazySamplers` builds the samplers four at a time. A `ParentSampler`
tabulates a parent's CDF on 20,001 points and inverts it by interpolation,
which is half a megabyte; the corpus has nearly 10,000 parents and the notebook
is already holding 20,000 fitted models, so building them all is not available.

### Every headline carries an interval

`plca.cluster_bootstrap` for a percentage or a median, `plca.nrmse_ci` for the
study's NRMSE, `plca.truth_win_share(rng=...)` for a win share. **The
resampling unit is the pLCA GROUP and never the row**, because the fifteen
method pairs inside a group share its materials and its variates and the four
materials share its total; `tests/test_plca.py` pins that a row bootstrap comes
back more than twice too narrow.

## 2bb. Which downstream metric the paper leads with

`src/metricset.py`, Stage 2g, called from the seven cells at the end of
notebook 3. The study's headline had always been ECI Rank #1 Frequency and
nothing had ever asked whether it is the right one.

**THE QUESTION CHANGED WHEN THE TRUTH RUN EXISTED.** Before Stage 2e a metric
could only be judged on whether it was stable, and "the six methods disagree
about this a lot" is not a reason to report it or to stop reporting it. With
every pLCA group run against its materials' TRUE parents on the same variates,
the question becomes whether a fitted model gets the metric right, and that is
what `recovery_table` measures.

**TWO STATISTICS, THE SAME DIVISION ON TWO NUMERATORS.** `plca.nrmse` is the
root mean squared difference BETWEEN two methods over the spread of the metric
across every material and method; `metricset.recovery_table`'s `recovery` is
the mean absolute difference between a method and the TRUTH over that same
spread. They are therefore directly comparable, and the pair says what neither
says alone: **a low NRMSE beside a high recovery error is a metric every method
agrees on and every method is wrong about**, which is the one thing a study
must not report as stable. `metric_verdict` is the join.

**THE DIVISOR IS THE TRUTH'S SPREAD AND NOT THE METHOD'S OWN.** Dividing by the
method's spread would let a method that reports nearly the same number for every
material improve its score by being less informative; `tests/test_metricset.py`
plants exactly that case and asserts the flat method loses. The truth's spread
is one number per (arm, metric) and is shared by all six methods, so the column
is a pure measure of error.

**AND THERE IS A SECOND DIVISOR IN THE SAME TABLE, WHICH IS A DIFFERENT
STATISTIC AND MUST NOT SHARE AN AXIS WITH THE FIRST.** `recovery` divides by the
SPREAD and answers "can this metric tell two materials apart under this method",
which ranks a CANDIDATE METRIC. `rel_error` divides by `truth_mean`, the LEVEL,
and answers "how wrong is this number", which is what compares one CLAIM with
another and is the only definition that can also be written for a building
total, a strategy's saving and a design comparison -- none of which has a
between-material spread at all. The two are not a fixed multiple of each other:
across the seven per-material outputs the level runs from 1.17 to 6.57 times the
spread, so a figure mixing them is not comparing like with like even within one
block of rows. Stage 2g's first scorecard did exactly that and decision 157
is the correction.

**`rel_error` IS A RATIO OF MEANS AND NOT A MEAN OF RATIOS**, so it is not the
ordinary mean absolute percentage error. The true uncertainty index reaches
-0.000671 and 2,904 of 60,000 materials carry a true value below a hundredth of
the mean, so a per-material ratio is unbounded and sometimes signless. Two tests
pin this. Measured against the alternative: on a material's mean contribution
the two forms agree, 12.63 against 12.88, and on the uncertainty index the mean
of per-case ratios is 134.81 against a ratio of means of 45.04 and a MEDIAN of
per-case ratios of 40.57 -- so the 135 is a few near-zero denominators, not
performance.

**`size_band_recovery` IS THE SAME STATISTIC SPLIT BY DATASET SIZE, and it is
the pooled scorecard's own caveat made measurable.** Counting which method is
closest on the most scorecard rows reads as a verdict between the families and
is not one: the corpus allocates 2,500 datasets to each of four size bands, so
half of every pLCA sits below 100 declarations, which is where a
three-parameter lognormal is already established to beat a kernel estimate.
Split by band the ordering inverts, and so does the weighting -- uniform
weights win below 100 declarations and market weights win above. It divides by
the output's true level over the WHOLE arm rather than within the band, so the
four rows of a column are on one scale; using each band's own level would make
a band with a smaller true value look better for free. Decision 161.

**THE WEIGHTING SCHEMES ARE RENAMED FOR DISPLAY AND NOT IN THE DATA.**
`fitting.WT_DISPLAY` maps Uniform to **"uniform weights"**, Variable to
**"market weights"** and the oracle scheme to **"known market shares"**, and
`fitting.display_method` applies it. Settled by the author 2026-09-25 after two
earlier attempts -- "Dirichlet shares" was accurate and inaccessible, "sampled
market shares" accurate and awkward -- and that churn is recorded in the
`WT_DISPLAY` comment so it is not repeated. "Variable" reads as "market shares
accounted for" and means "market shares drawn from a flat Dirichlet because
nobody publishes them", which is what made a result where uniform weighting
wins look like a modeling error; the label no longer carries that caveat, so
the METHODS SECTION does, at first use, and "known market shares" keeps the
drawn-versus-known contrast visible. Decisions 160 and 199. The
stored `method` values are UNCHANGED, because they are the join key between
every table this study writes and the eight regression fixtures. Stage 3 owns
carrying the display labels into the rest of the figures. Decision 160.

**`decision_agreement` IS THE SAME QUESTION WITH NO UNITS IN IT.** Every one of
these metrics is read as an argmax at some point -- which material is the
biggest -- and this is that reading scored against the right answer, with the
chance level `1 / k` printed beside it. It is the only comparison between two
candidate metrics that carries nothing of either metric's scale.

### The five questions, and why the scorecard is grouped by them

A probabilistic LCA answers five questions and every result this study reports
belongs to one of them: **magnitude**, what is the building's total; **attribution**,
which materials contribute most to it; **information**, which materials
contribute most to the UNCERTAINTY in it; **action**, how effective is a
reduction strategy; and **comparison**, is this design better than that one.
Stage 2e named them; Stage 2g's scorecard is the evidence for using them as the
paper's frame, and it is grouped by them.

**It is what makes the demotion of the rank metric legible.** "Which material is
the largest contributor" is one of six numbers inside ONE of the five questions,
not the study's subject. And the uncertainty index is the whole of the third
question rather than a footnote to the second.

**The cost of choosing a method, by question**, as the gap between the best and
worst of the six at the top of each question's range: attribution 35.5 percent,
action 31.0, magnitude 4.9, information 2.2, comparison 0.8. **The two questions
a designer acts on most directly are the two the choice affects least.**

### The magnitude companions

`eci_perc_mean`, each material's mean share of the building total, already
existed and no stage had compared it against the rank metric.
`plca.share_at_total_quantile` is new: each material's share of the total in the
iterations where the BUILDING sits at its 95th percentile, over a window of
plus or minus 0.01 in quantile units, which is 200 of the study's 10,000
iterations.

**IT IS NOT `eci_p95`.** That is the 95th percentile of a material's OWN
contribution taken over its own marginal, and the iteration that puts one
material at its 95th percentile is usually not the iteration that puts the
building at its 95th. A carbon budget is written against the building, so the
attribution question at the bad end has to read the shares where the building
actually is.

### The strategy rank frequencies, and the divisor

`strategy_rank_frequencies`. The cap rank frequencies carried a
`1 / (1 - capecc)` divisor, exact only while the cap was each method's own 75th
percentile and so bound in exactly 25 percent of iterations for every material
by construction; Stage 2e made the cap absolute and dropped the divisor with
it, leaving a plain count whose four columns summed to between 0.31 and 1.00.

**THE CORRECT DENOMINATOR IS THE ITERATIONS IN WHICH THE STRATEGY APPLIES**,
which is what the old divisor was reaching for and got only by assuming a
constant. `capecc_rank_1` now sums to exactly 1.0 across the materials of a
pLCA, like every other rank-1 frequency in the study, and the applicability is
REPORTED rather than divided away as `capecc_applies` and `capecc_binds`,
because under an absolute cap it is a property of the material and the method
and it is signal.

The ranking is among the materials whose cap BOUND. A material the cap does not
bind delivers exactly zero and any material it does bind delivers a strictly
negative change, so rank 1 is unambiguous; the lower ranks read "second best of
those that bound". Filling the non-binding materials with zero instead puts two
or three exact ties at the bottom whose averaged rank of 3.5 belongs to no
integer column, so those iterations would vanish from every column rather than
appear as the shortfall.

### The tail failure mode

`Contaminated` wraps any fitted model as a mixture with a narrow lognormal a
long way out and exposes the same pdf, cdf, ppf and `rvs_from_uniform`, so it
can be scored by the study's own W1 and sampled by the study's own pLCA without
either knowing it is not a fit. `tail_stress` replaces one material's model
with it, holds everything else, and reports what the contamination costs in W1
beside what it costs in every output; `tail_exposure` is the ratio.

**W1 TAKEN OVER THE SCORING GRID ALONE CHARGES FOR THE MASS AND NOT FOR THE
DISTANCE**, and that is asserted rather than described: the same contamination
weight at ten, a hundred and a thousand times the dataset mean gives the same
body score to within a part in a million, because above the grid's top the
integrand is clipped away. **The tail term Stage 2c added is what closes it**,
and the two are computed side by side in the notebook so the pair is visible.
A share and a rank frequency saturate and are immune; a mean, a standard
deviation and a variance share have no ceiling.

## 2c. Which characteristics carry signal

`src/metricreduction.py`, Stage 2f, called from **notebook 4**, which exists
only for it.

**IT IS ITS OWN NOTEBOOK, and that was worth doing.** The reduction reads three
tables the pipeline has already written -- the characteristics from notebook 1,
the scores from notebook 2, the run against the true parents from notebook 3 --
and needs none of the pLCA machinery. Kept inside notebook 3 it made a
15-minute analysis wait behind a 43-minute one, which is the wrong shape for
something still being iterated on. The run order is NB1, NB2, NB3, NB4.

**THERE ARE THREE TARGETS AND THEY ANSWER DIFFERENT QUESTIONS.** The LEVEL of a
method's own score, which is dominated by dispersion and dataset size because
every method gets worse on spread, small data. The CHOICE between two families,
which is `log(W1_KDE / W1_lognormal)` within a weighting scheme and is the
question the paper asks. And the ANSWER, how wrong the probabilistic LCA's
output is against the true parent. **Reporting the LEVEL ranking as an answer
to the CHOICE question is the mistake this module made in its first pass**;
decision 135 records it and `choice_frame` is the fix. Two properties of the
ratio matter: within a weighting scheme it CANCELS the definitional part of the
score, so `w_v_uw_wasserstein` is a legitimate predictor there and an identity
on the level, and it is scale free.

**23 candidates**: the 21 characteristics the old figure drew, plus the two
visible-mode counts, so that all three modality measures are separate
predictors. `METRIC_TRANSFORM` gives each one its modeling scale, because a
spline basis on a raw quantity spanning four orders of magnitude puts every
knot in the first percent of the range.

**EVERY RANKING IS OUT OF SAMPLE AND THE SYNTHETIC ARM IS PRIMARY.** The first
pass ranked on IN-SAMPLE incremental R2 with an F-test beside it, and both
halves were wrong for this question (decision 136). Adding a five-knot spline to
127 datasets raises in-sample R2 by about 0.043 UNDER THE NULL, and
cross-validated on those same 127 real categories the base model reaches an R2
of **-0.724** under market-share weights: it predicts worse than the mean, and
only 9 of 46 candidate gains exceed their own fold-to-fold spread. **No claim
about which method to use rests on the empirical arm.** The 10,000 synthetic
datasets are where the question is answered and the real ones are a check whose
interval is shown.

`cv_gain` reports a gain beside the standard deviation of that gain across
folds, and **no p-value appears anywhere in this module's output**. A gain
smaller than its own fold spread is not reportable however small its p-value
would have been. `forward_select` is the redundancy control: skewness, kurtosis
and the two normality statistics are four views of one shape, so a
one-at-a-time table credits the same effect four times, and selection admits a
candidate only if it still helps once everything already chosen is present.

**THE PRACTITIONER-FACING END.** `best_method_share` answers "is one method best
regardless" in size bands and `best_method_curve` does it continuously, on a
sliding window holding the same number of datasets at every position so the
curve does not get noisier in the sparse tail.

**WHERE THE FAMILIES CROSS, AND WHERE TO PUT THE RULE, ARE DIFFERENT
QUANTITIES.** `family_lead_curve` and `crossover_band` give the first: the size
range over which the leading FAMILY is genuinely in doubt, from the last size
one is clearly ahead to the first the other is, which is an interval of
ignorance rather than a confidence interval. On the corpus that is 46 to 70
declarations. `threshold_interval` gives the second, and it sits ABOVE the
crossing because the penalty curve is steeper on the high side.

**NO SINGLE-DECLARATION CUTOFF IS PUBLISHED, at the fit level or the claim
level (decisions 225 and 237).** The paper quotes two RANGES: **40 to 170**
declarations for the family split and **80 to 100** for the weighting split.
`mixedpolicy.MIXED_THRESHOLD = 80` is a constant the code needs to name one
policy, not a result. Any bare `81`, `68 to 97` or `68 to 106` found in this
repository outside the historical decision-log entries is stale; the decision
log keeps them because they are the record of what was measured.

**WHICH COMPARISON PRODUCES A CROSSING MATTERS MORE THAN THE CROSSING.** The
market-share kernel estimate passes the better SINGLE lognormal at 65; the
kernel FAMILY passes the lognormal family at 46 to 70; against the two
lognormals POOLED it does not pull clear until past 200. Quoting one without
naming the comparison is how a figure ends up disagreeing with its own caption,
which happened here.

`threshold_interval` runs two bootstraps over datasets. One resamples, refits
the cost curve and takes its argmin, so its spread says how well the data pin
the threshold down. The other is PAIRED -- each threshold's excess over
whichever won on that same resample -- so common variation cancels and the
interval is about the difference. **Use it rather than `flat_region`**, which
answers a nearby question with a tolerance chosen rather than measured and no
uncertainty at all, and which reported a band nearly twice too wide at the top.

`longest_true_run` exists because a figure took the min and max of the
indistinguishable flag instead of its longest unbroken run: at the edge of a
near-zero effect the flag jitters, 106 excluded at a lower bound of 0.120 while
116 is included at 0.000, and one isolated threshold widened the reported band
by nearly a factor of two at the top. The band that rule was guarding is a
fit-level one and is no longer published; see the note above.

`effective_sample_fraction` and `weighting_by_concentration` answer when
market-share weighting pays, split by how concentrated the weights are INSIDE a
size band so the split is not dataset size under another name.

`flat_region` is kept for the question it does answer -- how flat the cost curve
is -- and reads the SPAN rather than the argmin, measured against what the rule
is worth rather than against the best cost, because a relative tolerance
collapses onto a single point as the best cost approaches zero, which a test
caught.
**ALL THREE ARE SCORED AGAINST `w1_market` AND THE TARGET IS DOING REAL WORK.**
Under `w1_parent` each weighting scheme is graded against a different
population; under the in-sample `w1` every model is scored against the
variable-weighted data. The variable kernel fit beats its uniform twin above
n = 1,000 on 75.8 percent of datasets under `w1_market`, 20.7 percent under
`w1_parent` and 98.2 percent under `w1`. Only the first answers the question.

**Two model families, one instrument.** A penalized additive model (natural
cubic splines per predictor, elastic net) and gradient boosting, both scored
out of sample and both ranked by permutation importance on the HELD-OUT fold,
so the rankings are comparable. `rank_survivors` refuses to rank a model whose
out-of-sample R2 is below `min_r2`, because inside such a model the importance
ordering is noise.

**Missingness is handled rather than defaulted.** The additive model imputes
with an indicator and the boosted model splits on missingness natively, so both
keep every row. `complete_case_cost` reports what dropping the rows would have
cost, per arm and size band, which is the number that says why.

**The tautology guard.** `definitional_check` reports each candidate's rank
correlation with `w1_definitional`, the part of a fit score no estimator can
remove. `w_v_uw_wasserstein` reproduces it EXACTLY for all three
uniform-weighted methods, so ranking it first on a fit target is an identity
being rediscovered; `DEFINITIONAL_CANDIDATES` names it and every survivor
ranking is reported with and without it. On the downstream error no such
identity exists.

**The curves that replace the rolling averages.** `binned_curve` gives
equal-count bins with a within-bin percentile bootstrap and the count on every
row; `lowess_curve` gives the smooth read through them. The old rolling-average
figure is kept beside the new one so the two can be checked against each other.

**THE BAND COMES FROM THE BINS AND NOT FROM THE SMOOTHER, and that is a cost
decision made on a measurement.** statsmodels' LOWESS runs three robustifying
iterations by default; at 10,000 datasets one fit takes about 600 ms, so
bootstrapping the smoother 400 times over five characteristics, six methods and
three targets is some 12,000 fits and several hours. Turning the iterations off
is 255 times faster and is NOT available: it moves the curve by 54 percent of
its own range on the target's scale and 14 percent on the log scale, six times
the width of the band it would be drawn inside. So the binned bootstrap carries
the uncertainty -- it is exact, cheap and reports its own counts -- and the
smoother is fitted once with the iterations intact. The whole synthetic arm
takes 17 seconds.

**The density rug is a DENSITY.** The bins hold equal counts, so a bar of
constant height would tile the axis and say nothing; the height is the count
over the bin width, so a narrow bin stands tall.

---

## 2d. The robustness sweeps, and what they settled

Stage 2h. Every item here closes a "you only tested one variant" objection, so
each is a sweep with a tabulated result under `outputs/tables/audits/` rather
than a one-off check. Decision 56 permits audit scripts to write there and
nowhere else in `outputs/`.

### One market-share rule for both arms

`weighting.coherent_weights`. Until Stage 2h the two arms drew market shares by
DIFFERENT rules on the dimension the paper is built on: a flat Dirichlet over
individual declarations on the real categories, a share per mixture component
split inside the component on the synthetic ones. Independent weights are
exchangeable, so the weighted CDF converges to the unweighted one and the
measured effect MUST decay like n^-1/2 whatever markets do.

The rule is the synthetic arm's, ported. Cut the declarations into k groups,
draw each group's share from a Dirichlet, split it inside the group. Groups
stand in for the components real data does not label, and the cut is a
contiguous run of the SORTED values, ordered by

    s = rho * rank + (1 - rho) * uniform

so `rho` is the coherence axis and `k` the concentration axis. Decision 97
requires the two be separated and this is how.

**A MIXTURE MODEL WAS REFUSED for the proxy**: it cannot be estimated at three
to nine declarations, mode counts on real data swing from 95 to 68 percent
unimodal on one smoothing choice, and it would put a fitted model inside the
paper's central quantity.

**THE BLOCK COUNT IS `draw_blocks`, uniform on 1 to 5 and INDEPENDENT of n**,
because that is what `genconfig.k_min` and `k_max` do. A first version grew it
with n up to 12; that is a different weight model, and it reintroduces the
artifact the port exists to remove. `blocks_for_n` is kept only so a sweep can
measure what the wrong rule costs.

**THE PROXY IS VALIDATED AGAINST THE TRUE MODE LABELS**, which only the
synthetic arm can do because only it has both. At `rho = 0.5` a contiguous cut
reproduces the true-label weighting effect to within 4 percent on the typical
dataset, and orders the categories most like the truth (Spearman 0.935). That
is the anchor for the port.

**rho = 0 IS NOT THE NEUTRAL CHOICE.** It is the claim that market share is
uncorrelated with carbon intensity, which published production volumes
contradict. A bigger separation is not evidence of a better model either; the
separation measures what unknown shares do and is not a target.

**NOTHING IS REWEIGHTED IN THE PRODUCTION PATH.** The rule, its validation and
its sweeps are measurements. Applying it would move every weighted
characteristic of the empirical arm, which the generator is calibrated against.

### The judgment arm

`src/judgment.py`. The pedigree matrix as a lognormal specified by a center and
a geometric standard deviation, plus a uniform and a triangular over a
plausible range derived from the SAME two inputs, so the three differ in SHAPE
and not in information. All three expose the same pdf/cdf/ppf/rvs_from_uniform
interface, so the truth run, the scoring and the pLCA take them unchanged.

**THEY ARE NOT IN THE MAIN COMPARISON AND THAT IS A DECISION ABOUT WHAT THEY
ARE FOR.** A uniform fitted to n declarations is exactly their smallest and
largest and nothing else, so it would lose a goodness-of-fit comparison by a
distance that says nothing. Decision 151. What makes the comparison possible at
all is the yardstick: the error against the TRUE distribution does not care how
a model was built.

**TWO DIMENSIONS, AND THE MODE OF THE OFFSET MATTERS MORE THAN ITS SIZE.** A
displacement applied identically to every material cancels EXACTLY in a design
comparison, because both options' totals scale by the same factor; one drawn
per material does not. `OFFSET_MODES` separates them, and sweeping only the
common case would have reported a null that was an artifact of the sweep. The
primary location model has no free parameter: the center is ONE declaration
drawn at random, which is what a practitioner without a dataset holds.

**THE SPREAD IS SWEPT RELATIVE TO THE DATA'S OWN.** The pedigree matrix's
factor table is not in `refs/`, and this project has already had to withdraw a
figure quoted from memory (decision 49's amendment), so no table is asserted.
`gsd_ratio` answers decision 124's deliverable sentence without one. **The
manuscript owes a sourced factor table before laying a specific pedigree score
on this axis.**

### The upper truncation

`families.Truncated` takes an optional `hi`, defaulting to infinity and
verified bit-identical by the regression fixtures. `TruncatedAbove` caps any
fitted model, including the kernel estimate, and `cap_models` applies a cap
read off the data as a multiple of the largest observation, which is the only
anchor available once every dataset is normalized to a mean of 1.0.

**THE TAIL TERM STAYS IN FORCE THROUGHOUT THE SWEEP.** Without it the criterion
returns the same number to seven significant figures whether the misplaced mass
sits at a hundred times the dataset mean or a thousand, so the sweep would be
measuring nothing. Decision 149.

### Two more data-driven families

`weibull` joins `gamma` in `fitting.FAMILIES`. Both are natively on (0, inf)
with no threshold, so the renormalizing constant is exactly 1 and decision 13's
support costs them nothing. **`_fit_w1` keeps a named list of parameter keys on
the way back to the builder and the Weibull shape 'c' was not on it**, which
dropped the shape silently rather than raising; a test pins the round trip.

### The scorecard's numerator

`metricset.claim_scorecard` and `per_unit_error`. Stage 2g's decision 157 put
every scorecard row on one DIVISOR; this puts every row on one NUMERATOR.

    total_error      mean |error| PER UNIT over the true level. The error in a
                     SINGLE decision, and what the figure draws.
    portfolio_error  |mean error| over the true level. The error in the AVERAGE
                     claim over many decisions, which is the right quantity for
                     a stock model and the wrong one for one design.

Both are kept for all fifteen rows because they are different questions, not a
better and a worse answer.

### How many digits of a fitted crossing to print

`flip.prose_digits`, `prose_crossing`, `crossing_precision`. Every crossing this
project publishes is inverted from a FITTED curve, and the bootstrap interval
around one fit says how well the data pin down THAT curve, not whether the
curve is the right shape. The isotonic fit falls outside the logistic interval
on five of the six published constants.

**PROSE AND FIGURE ANNOTATIONS print the significant figures the two fits agree
on plus the first they part at, and print BOTH values where they still differ
there. Result tables and the supplement keep both fits and the interval at FULL
precision.** `flip.FLIP_THRESHOLDS` is a computational constant rather than
prose and is untouched.

## 2e. Letting the method vary by material

`src/mixedpolicy.py`, Stage 2j, called from eight cells at the end of notebook
3. Every probabilistic LCA above those cells fits ONE method to all four of its
materials; this asks what happens when it does not.

**THE RULE IS ONE NUMBER AND IT IS THE STUDY'S OWN.** A kernel estimate with
market weights at or above `MIXED_THRESHOLD = 80` declarations, a
three-parameter lognormal with uniform weights below. **80 from 2026-09-30, decision 224**, and it
is a round reference constant the code needs in order to name one policy, NOT a
published result. **The paper quotes two numbers and both are ranges**: 40 to
170 declarations for the family split and 80 to 100 for where knowing market
share starts helping. No single-declaration threshold is published. **The rule the paper recommends
is the FEASIBLE one** -- uniform weights throughout, only the family switching
-- because nobody publishes market shares (decision 216), and its published
cutoff range is **40 to 170** (decision 224). `select_method` takes exactly one required argument and a test
asserts that, because a rule with two numbers in it is not the deliverable and
decisions 88 and 139 tested every other characteristic without finding one that
yields a usable threshold.

**NOTHING IS REFITTED.** `add_mixed` puts a NEW KEY on each dataset's existing
`models[dataset]` dictionary pointing at whichever of the six already-fitted
model objects the rule selects, so the object SCORED and SAMPLED under the
mixed policy is the same object that fixed policy uses. A group whose four
materials all sit on one side of the threshold therefore reproduces that fixed
policy bit for bit, which is the control the notebook prints and two tests pin.

**TWO RULE FAMILIES, AND ONLY ONE IS A METHOD.** A practitioner can never
know market shares, so uniform weighting is not a choice they make -- it is
the situation they are in. `feasible_policies` is therefore the rule the paper
recommends: uniform weights throughout, the FAMILY switching at the cutoff.
`sweep_policies` adds the TRUE market shares above the cutoff and is not
something a reader can follow; the gap between the two curves is what knowing
market share would be worth, which is about 3 points of the roughly 23 a
probabilistic LCA gets wrong. `threshold_curve` takes a `family=` argument and
REFUSES a curve over both at once, because the two are swept on the same grid
and a combined curve would have two points at every cutoff.

**WHAT THE MARKET WEIGHTS ARE, because three stages described them wrongly.**
On the synthetic arm they are the TRUE market share of every product group, to
1.1e-16; only the division of a group's share among the products inside it is
arbitrary. So uniform against market on this arm is IGNORING a known share
against USING it, never guessing against knowing. Decision 212;
`tests/test_mixedpolicy.py` pins it against the generator.

**THE CUTOFF IS SWEPT, NOT ASSERTED.** `all_policies` builds thirteen cutoffs
from 3 to 10,000 for EACH family, plus three one-axis variants that say which
half of the known-share switch does the work -- and it is the weighting, which
is the half a reader cannot use (decisions 209, 216). **The two degenerate ends
make the sweep self-checking**: the corpus holds 3 to 9,999 declarations, so at
a cutoff of 3 each rule IS its own above-method and at 10,000 its own
below-method, and all four ends must reproduce a fixed method exactly. The
notebook prints that check. `threshold_curve` is the claim-level twin of
`metricreduction.threshold_interval`: two bootstraps over
pLCA groups, the second paired against whichever cutoff won on that resample,
and the longest UNBROKEN run of indistinguishable cutoffs is reported beside
the span from the lowest to the highest such cutoff. **The two differ when an
INTERIOR cutoff falls out on bootstrap jitter**; the run rule was written to
stop a lone far-away point widening a band and is the wrong instrument for a
hole in the middle. Every candidate is still ONE number on ONE input and a test asserts it.

**POOLING RUNS OVER TWO UNIT UNIVERSES.** Fourteen of the fifteen claims belong
to a pLCA GROUP and the design comparison belongs to a design PAIR from its own
resampling. `claim_blocks` builds one array per universe and each is resampled
in its own; a single array indexed by pLCA group leaves the design comparison
as a column of NaN, so a number described as pooled over fifteen claims would
quietly be over fourteen.

**THE STAGE RUNS ITS OWN TRUTH PASS rather than extending the study's**, and a
later stage adding an eighth policy should do the same. A win share, a
`best_method` and a `stakes` are properties of the SET of policies compared, so
adding a seventh to the existing run would change every one of them in the
six-method tables the paper reports, for a reason that has nothing to do with
any method changing. The cost is about sixteen minutes of the notebook's
110; the benefit is that re-running notebook 3 reproduces every
pre-existing table content-identically.

### The comparison is paired, and the pivot that makes it so

`claim_errors` returns one row per (claim, unit, policy) and `_wide_units`
pivots it on an explicit UNIT key. **Pivoting on the frame's own index instead
leaves one value per row, so a row minimum returns that row's own value and the
per-material oracle comes back as the grand mean over all policies -- worse
than the best fixed one, which an oracle cannot be.** That was the first draft
and `tests/test_mixedpolicy.py` is where it is written down.

`_paired_boot` is the same estimator as `plca.cluster_bootstrap` -- whole pLCA
groups resampled, the statistic the mean over the rows they carry -- written as
a ratio of per-cluster sums so that fifteen claims over 2,500 clusters at 2,000
resamples costs seconds rather than minutes. A test pins that the two agree.

`claim_gain` compares against the **best FIXED policy on that claim** by
default, which is the comparator a reader would otherwise use and therefore the
hard test; `oracle_ceiling` adds the unreachable per-material minimum as a
BOUND and never as a policy, and it is optimistic by construction because a
minimum over six noisy errors is biased low.

### A design pair is not a pLCA group, and their ids collide

`claim_errors` records `cluster_kind` and `attach_composition` joins only rows
whose cluster really is a pLCA group. The design comparison's cluster is a
design PAIR from its own resampling and its ids run 0 to 2,499 exactly as the
pLCA groups' do, so a join on the id alone hands every pair an unrelated
group's composition. **It surfaced as a cell in which the rule cannot act
reporting a two percent gain**, and it was invisible in every aggregate.

## 3. Seeding and caching

**Seeding.** All randomness comes from an explicitly passed
`numpy.random.Generator`. No function in `src/` creates a Generator or touches
global numpy state. Each notebook creates exactly one:

```python
SEED = 20260911
rng = np.random.default_rng(SEED)
```

`tests/test_notebooks.py` enforces this: exactly one `np.random.default_rng`
per notebook and no other `np.random.*` call anywhere. A figure that needs its
own stream spawns it, as the three strip plots in notebook 2 do -- seaborn's
`stripplot` was drawing its jitter from the global state, so the same table drew
a different figure on every run.

The same file guards three other properties a reader of the deposit depends on:
notebooks carry NO stored output (strip them with `jupyter nbconvert
--clear-output --inplace notebooks/*.ipynb`, and re-run to see results), no
notebook exceeds 8 MB, and nothing is saved above 300 dpi. A notebook that sets
`figure.dpi` must set `savefig.dpi` beside it, because `savefig.dpi` defaults to
`'figure'` and a high screen resolution silently becomes the file resolution. `tests/test_determinism.py`
enforces that generation is reproducible from the seed and neither reads nor
advances the legacy global stream.

Record the seed in the run-metadata file beside any table the notebook writes.

**The parent of a synthetic dataset is RECOVERED, not read.** `parents.json.gz`
stores how each parent was asked for: each component's moment TARGETS, from which
`components.solve_component` recovers its location and scale deterministically,
plus the global shift and the truncation bounds. It does NOT store the
displacement the overlap solve gave each component, and one recorded overlap
value cannot identify k - 1 displacements, so the parent CDF cannot be written
down from the record. An earlier version of this file claimed it could.

`corpus.rebuild_parents` replays the generation loop instead, which is
deterministic given the seed, and keeps the parent objects `generate_corpus`
discarded. It is not a regeneration: no corpus is written and nothing is
redrawn. It refuses unless `genconfig.DEFAULT` still equals the configuration
recorded in the corpus, checks twelve record fields per dataset, and compares the
replayed values and weights against `values.parquet` element by element. On
every corpus this has been run against, including the current
`corpus_2026-09-25`, all 10,050 datasets replay byte-identically, in about 13
minutes.

**IT REFUSES A CORPUS GENERATED BEFORE `genconfig` GAINED A FIELD**, because it
compares configuration dictionaries and is right not to reason about which
differences are inert. When the specs are already cached, load them rather than
replaying: `audits/draft_end_to_end.py` shows the pattern and explains why that
is not a weakening of the check.
The result is cached as `parents_spec.json.gz` inside the corpus directory;
`corpus.load_parent_specs` and `load_parent_objects` read it and build it if it
is absent.

```bash
python -c "import sys; sys.path.insert(0,'src'); import corpus; corpus.rebuild_parents()"
```

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

**Recomputing the characteristics is NOT regeneration, and there is a separate
entry point for it.** The empirical arm computes its characteristics when
notebook 1 runs; a corpus stores them at generation time. A correction to the
metric code therefore reaches one arm and not the other unless the corpus is
told to re-read its own values:

```bash
python -c "import sys; sys.path.insert(0,'src'); import corpus; corpus.remetric_corpus('<new-label>')"
```

`remetric_corpus` copies `values.parquet` byte for byte, reruns the current
`empirical_metadata` over it, and carries the generation record across. No
dataset is redrawn and no random number is consumed. Use this, never
`generate_corpus`, when a metric is corrected; generation itself is closed.

Each corpus directory holds:

| File | What |
|---|---|
| `values.parquet` | long format, `dataset_id, value, weight` (decision 15) |
| `metrics.parquet` | one row per dataset: stratum, metrics, generation record |
| `parents.json.gz` | how each parent was ASKED for: component moment targets, the overlap target, the shift, the bounds. **NOT enough to rebuild its CDF**; see below |
| `parents_spec.json.gz` | the finished parent of each dataset, written by `corpus.rebuild_parents`. Derived, and the file the recovery score reads |
| `mode_labels.parquet` | which mixture component each stored value was drawn from, written by `corpus.rebuild_mode_labels`. Derived the same way and for the same reason -- the parent shuffles the points it draws, so nothing else records it -- and read by the oracle-weight counterfactual. About 4.5 MB; the replay that builds it takes 14 minutes and happens once per corpus |
| `combos.csv` | the 2,500 disjoint pLCA groups of four |
| `runmeta.json` | seed, full config, git commit, library versions, platform, counts |
| `invalid_datasets.json` | what the validity filter rejected, and why |

## 4. Running

```bash
conda env create -f environment.yml
conda activate compareuq
python -m ipykernel install --user --name compareuq --display-name compareuq
python -m pytest tests/          # 632 tests, 0 skipped, about 177 s
                                 # it FELL in Stage 4 and the drop is
                                 # exact: three tests run per report,
                                 # six handoffs went to the retention
                                 # rule, and the surviving report adds
                                 # three cases back. 642 -> 627
```

Headless execution, from `notebooks/`:

```bash
python -m nbconvert --to notebook --execute \
  --ExecutePreprocessor.kernel_name=compareuq \
  --output-dir=/tmp/nbrun --output=out.ipynb 03_CompareUQ_PerformPLCA.ipynb
```

**Smoke configuration.** Notebook 3 is the expensive one. Set
`COMPAREUQ_SMOKE_COMBOS=20` to run it on 20 pLCAs instead of 2,500, which
exercises the pipeline end to end in well under a minute.

**A SMOKE RUN CAN NO LONGER REACH `outputs/`, as of Stage 2e.** Every path
notebook 3 writes goes through `OUT`, which smoke mode points at a fresh
temporary directory, so the repository is untouched and `git checkout --
outputs/` is no longer the thing standing between a smoke run and a commit. Two
tests hold it in place: `tests/test_notebooks.py` refuses a literal `outputs/`
path in any cell of notebook 3 and checks that `OUT` is defined before the first
write, and `tests/test_plca.py` asserts that the committed pLCA table is a full
run -- not flagged as a smoke run in its metadata, at least 2,000 groups, and
with the row count that metadata implies.

What it is protecting against happened in Stage 2d: a smoke run reached a commit,
replacing the 60,000-row pLCA table with a 960-row one and redrawing seven
figures from 40 groups instead of 2,500. The table was spotted because its damage
showed as a row count; the figures were not, and came back only when the notebook
was rerun in full. Discrepancy entry 87.

```bash
COMPAREUQ_SMOKE_COMBOS=20 python -m nbconvert --to notebook --execute ...
```

Use it before any full run. It caught two defects in Stage 1 that had
previously only surfaced eleven minutes into a full execution.

**AND VERIFY THAT AN EDIT LANDED BEFORE STARTING A LONG RUN.** In Stage 2e a
patch script hit an assertion on a string a previous edit had already changed,
so it exited before writing the notebook; because it shared a command line with
a backgrounded `nbconvert`, the failure was invisible and the 45-minute run that
followed faithfully reproduced the unedited notebook. Grep the notebook for the
new text before launching. This costs two seconds and the alternative costs an
hour.

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

**Approximate runtimes, re-measured end to end in Stage 3 and corrected by
decision 240, with NB4 corrected again in Stage 4 from 20 to 15 after it ran
in 13.6 minutes: NB1 about 35 min, NB2 about 15 min, NB3 about 110 min at
`neccs = 10000`, and NB4 about 15 min.** They are wall clock with nothing else
competing; a machine running other work can take half again as long, which is
why they are quoted to the nearest five minutes and not defended further. Three
different sets of these numbers were in the repository before Stage 3's review
-- this file, the README and the Stage 3 report each carried its own -- and one
set now stands in all three places. The paragraph below is the Stage 2f
profile, kept because it says WHERE the time goes, and its totals are
superseded by the line above.
Profiled cell by cell in the Stage 2f review, which also corrected a wrong
figure: NB3 was reported as two hours, which was wall clock while other jobs
competed for the processor, against 58 minutes of actual cell time -- 43 of it
the pLCA and its sweeps, 15 the metric reduction, which then moved to NB4. The
single most expensive cell in the project is NB3's crossed sweep over group
size and material use intensity at 13.5 minutes, which is 23 percent of that
notebook. NB1's figure was recorded as
3 min through Stage 2c and has been wrong since Stage 2d added the per-dataset
weighting risk at 1,000 Dirichlet draws, which is almost all of it; NB3 grew
again in Stage 2f, which added the metric reduction. The Stage 2e figures,
which the rest of this paragraph describes, were measured end to end in Stage
2e, which
roughly trebled it by adding the sweep, the flip recalibration at six group
sizes, the run against the true parents, the design comparison and the
oracle-weight counterfactual. **The first run on a new corpus adds 14 minutes**
for the mode-label replay, which is then cached. All three roughly doubled in Stage 2b,
because stratum 4 now reaches n = 9,999 where the pre-regeneration corpus
stopped at 749, and NB2 grew again in Stage 2c: the whole corpus is scored
against its parent and the empirical arm is cross-validated at ten repeats.
`COMPAREUQ_SMOKE_COMBOS` applies to NB3 only.

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

The analyzed set is the corpus minus the probe set: **9,999 datasets, not
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
`audits/build_raw_extract.py`, which refuses to overwrite an
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
| `TABLE_MethodScores.csv` | NB2 | one row per (arm, dataset, method): W1, held-out W1, both ranks, model-spread ratio |
| `TABLE_MethodSummary.csv` | NB2 | the six methods by arm, the table to read first |
| `TABLE_MethodCurves.parquet` | NB2 | every score against every characteristic, unbinned, with the rolling mean the figures draw. **Parquet since Stage 3**: 4.0M rows of which five columns are repeated strings, 96.4 MB as csv.gz against 44.4 as parquet and 23x faster to read. Every float is still float64 |
| `TABLE_BandwidthRules.csv` | NB2 | the two KDE methods under all three bandwidth rules |
| `TABLE_TargetComparison.csv` | NB2 | **the Stage 2c table to read.** One row per (arm, dataset, method): the in-sample score, the recovery score against the parent it estimates and against the market parent, the cross-validated score with its spread across splits, the decomposition and the overlap area |
| `TABLE_TargetSummary.csv` | NB2 | the six methods by arm, criterion and weighting scheme |
| `TABLE_CrossValidatedScores.csv.gz` | NB2 | one row per (dataset, method, split, direction) |
| `TABLE_CrossValidatedSummary.csv` | NB2 | the mean over splits and the spread across them |
| `TABLE_PairedBootstrap.csv` | NB2 | whether a gap between two methods survives resampling the datasets |
| `TABLE_WeightingDecomposition.csv` | NB2 | fit error against the definitional gap |
| `TABLE_WeightingOnCommonTarget.csv` | NB2 | do market weights help, on the market parent, by size band. Its columns are `uniform`, `market` and `market_wins`; they were `variable*` until Stage 4 |
| `TABLE_Regret.csv` | NB2 | mean, median and upper tail of regret per method |
| `TABLE_PostStratifiedScores.csv` | NB2 | every headline aggregate equally allocated and reweighted. **NOT `TABLE_PostStratified.csv`, which is NB1's and is about the dataset characteristics** |
| `TABLE_ModalityConditioned.csv` | NB2 | the method comparison split by visible modality, within size band |
| `TABLE_PolicyComparison.csv` | NB2 | **the table a practitioner reads.** Each fixed and size-conditional policy against the per-dataset oracle: mean cost, share within 5 and 20 percent of the best, and the worst single dataset |
| `TABLE_RuleSelection.csv` | NB2 | whether adding a characteristic to the rule beats a size threshold alone. It does not |
| `TABLE_RuleCandidates.csv` | NB2 | how much each characteristic adds to predicting the KDE-lognormal gap once log(n) is in the model |
| `TABLE_SizeCrossover.csv` | NB2 | the fitted slope and break-even n for each arm |
| `TABLE_MaterialTiers.csv` | NB2 | every category with its material tier and size. **Publish this**: a hot-spot argument cannot be checked without it |
| `TABLE_MethodByMaterialTier.csv` | NB2 | the method comparison inside each tier |
| `TABLE_CharacteristicsByTier.csv` | NB2 | median characteristics by tier, which is why the tiers differ |
| `TABLE_MaterialTiers.csv` | NB2 | every category with its material tier. **Publish this**: a hot-spot argument cannot be checked without it |
| `TABLE_MethodByMaterialTier.csv` | NB2 | the method comparison inside each tier, and for structural categories at n >= 100 |
| `TABLE_VisibleModes.csv` | NB1 | visible modes per dataset at scipy's default bandwidth and at the one the study fits |
| `TABLE_VisibleModeSummary.csv` | NB1 | the share with one, two, three or more visible modes, at both bandwidths |

**Figures added in Stage 2g: ONE, `FIG_ClaimScorecard`.** Fifteen claims by the
six methods and, from Stage 3, the size rule under the five questions, every cell the method's own distance from
the truth on one definition, with a bar beside it for what the choice of method
costs. **Two others were built and cut in the same stage.** `FIG_MetricChoice`
showed the same recovery error as a best-to-worst range, hid which method was
which, and the scorecard says everything it said. `FIG_TailBlindSpot` showed the
contamination sweep; it is a stress test rather than an observation, the guard
already bounds the failure mode, and Stage 2h's upper truncation removes it, so
the finding is two tables and a paragraph. Decisions 153 and 158.

**Figures added in Stage 2f:** `FIG_CharacteristicSurvivors_Empirical` and
`_Synthetic`, which are what the 21-panel characteristic figure becomes;
`FIG_MarginalVersusPartial`, the marginal view above the multivariate one,
which is the stage's claim in one panel; and three supplements,
`SUPP_CharacteristicSurvivors_Answer`, `SUPP_MarginalVersusPartial_Empirical`
and `SUPP_RollingVersusBinned_*`, the last of which keeps the rolling average
beside its replacement so the two can be checked against each other.

**Figures added in Stage 2c:** `FIG_EvaluationTarget`, `FIG_TargetBySize`,
`FIG_Regret`, `FIG_MethodByMaterial` (which is the POLICY comparison, not a
material breakdown -- the tier is not a mechanism, decision 84) and
`SUPP_AllEmpiricalFits`, all 147 empirical datasets with all six fits.
| `TABLE_MethodWinShare.parquet` | NB2 | how often each method wins, against the percentile of each characteristic. Parquet since Stage 3, 22.8 MB to 6.7 |
| `TABLE_WeightingLocationShape.csv` | NB1 | one row per (arm, dataset): the uniform-to-variable W1 split into the mean shift it must at least contain and the residual |
| `TABLE_WeightingRelativeMeasure.csv` | NB1 | the same quantity computed on normalized and on RAW values, which is the check that the normalization is not doing secret work |
| `TABLE_WeightingRisk.csv` | NB1 | **the per-dataset practitioner table.** A_IQR, the median and 90th percentile separation over 1,000 Dirichlet draws, and the probability that a possible weighting carries a 1, 5 or 10 percent chance of changing the top contributor |
| `TABLE_WeightingRiskDrivers.csv` | NB1 | Spearman correlations that separate A_IQR, which follows dataset SIZE, from the risk, which follows DISPERSION |
| `TABLE_WeightingRiskPostStratified.csv` | NB1 | both measures at equal allocation and reweighted to the empirical size mix |
| `TABLE_FlipNoiseFloor.csv` | NB3 | the flip rate with IDENTICAL models under two independent streams. **Read this before any other flip table** |
| `TABLE_FlipMethodPairs.csv.gz` | NB3 | one row per (pLCA, method pair): how far apart the two fitted models are, and whether the answer changed |
| `TABLE_FlipCalibration.csv.gz` | NB3 | the calibration set: uniform weights against tempered Dirichlet weights, at nine levels |
| `TABLE_FlipCrossings.csv` | NB3 | **the stage's deliverable.** The relative W1 at which the flip probability crosses 1, 5 and 10 percent, with bootstrap intervals and an isotonic comparison |
| `TABLE_FlipCurve.csv` | NB3 | the observed flip rate in equal-count bins, which the figure draws |
| `TABLE_FlipProvenance.csv` | NB3 | whether the curve describes the distance or where the distance came from |
| `TABLE_FlipPostStratified.csv` | NB3 | the flip rate at equal allocation and reweighted |
| `TABLE_PLCAResults.csv` | NB3 | 59,976 x 43, which is 2,499 groups x 6 methods x 4 datasets |
| `TABLE_PLCAResults_runmeta.json` | NB3 | seed, neccs, versions, platform, and whether the run was a SMOKE run |
| `TABLE_CRNComparison.csv` | NB3 | **read this before any statement about how far apart two methods are.** Each output's median change between two methods under shared variates, under independent variates, and under one method run twice |
| `TABLE_CRNComparisonRows.csv.gz` | NB3 | the rows behind it, one per (pLCA, output, kind, method pair) |
| `TABLE_PLCASweep.parquet` | NB3 | the crossed sweep: one row per (cell, pLCA, method pair), with the top-two contribution ratio and the leading material's share. **Parquet and not CSV**: 432,000 rows of mostly floats, 39 MB gzipped as CSV against 33 MB columnar, written this way for the typing and the read speed as much as the 16 percent saving. Decision 15 already settled parquet for this project's large tidy tables |
| `TABLE_PLCASweepSummary.csv` | NB3 | one row per (group size, intensity case): the flip rate and each output's shift, each with a bootstrap interval |
| `TABLE_PLCAGroupSize.csv` | NB3 | the equal-intensity column of that sweep, which is how the effect of choosing a UQ method scales with the number of materials |
| `TABLE_PLCARatioCrossings.csv` | NB3 | **the intensity sweep's deliverable.** The top-two contribution ratio at which the flip probability crosses 1, 5 and 10 percent, pooled and by group size |
| `TABLE_PLCARatioCurve.csv` | NB3 | the observed flip rate in equal-count bins of that ratio, with the continuous outputs in the SAME bins, so the figure's two panels are read off one binning |
| `TABLE_PLCARatioAnchor.csv` | NB3 | the one real top-two contribution ratio available, transcribed from the text of Marsh et al. (in press) |
| `TABLE_FlipCrossingsByGroupSize.csv` | NB3 | the Stage 2d flip thresholds recalibrated at 2, 3, 4, 6, 8 and 12 materials |
| `TABLE_FlipCalibrationByGroupSize.csv.gz` | NB3 | the calibration rows behind it |
| `TABLE_PLCATruth.csv.gz` | NB3 | **the Stage 2e table to read.** One row per (pLCA, material, method, truth parent): every output, the value the TRUE parent gives, and the error. **The SIX methods only** |
| `TABLE_PLCATruthRule.csv.gz`, `TABLE_PLCATruthBuildingRule.csv.gz`, `TABLE_PLCATruthInterventionRule.csv.gz`, `TABLE_PLCADesignSwapRule.csv.gz` | NB3 | the same rows for the FEASIBLE size rule, which rides along on the main truth and swap passes so the seven-policy scorecard is one experiment rather than two (Stage 4, decision 237). They are separate files so the four tables above keep exactly the rows and the order they had |
| `TABLE_PLCATruthSummary.csv` | NB3 | per method, the mean absolute error against the truth with an interval, and how often it names the true largest contributor |
| `TABLE_PLCATruthWinShare.csv` | NB3 | how often each method is closest to the truth, with an interval |
| `TABLE_PLCATruthPostStratified.csv` | NB3 | the same error at equal allocation and reweighted to the empirical size mix |
| `TABLE_PLCATruthByIntensity.csv` | NB3 | the same error at 1:1, 2:1 and 10:1 intensities, which is what crosses the truth run with the dominance sweep |
| `TABLE_PLCATruthBuilding.csv.gz` | NB3 | statement 1, per (pLCA, method): W1 and the Cramer distance from the method's whole-building total to the true one, the error at two quantiles, and the error in the compliance statement |
| `TABLE_PLCABuildingSummary.csv` | NB3 | that, per method, with a bootstrap interval |
| `TABLE_PLCATruthIntervention.csv.gz` | NB3 | statement 4, per (pLCA, material, method): what capping or a quantity reduction delivers, the chance it delivers at least 5, 10 and 20 percent of the building, and the error in each |
| `TABLE_PLCAInterventionSummary.csv` | NB3 | that, per method, with a bootstrap interval. **The share of iterations capped is in here and is no longer 25 percent by construction** |
| `TABLE_PLCADesignSwap.csv.gz` | NB3 | statement 5, per (pair, saving, method): the discernibility index and the modified comparison index against the truth |
| `TABLE_PLCADesignSwapSummary.csv` | NB3 | **the stage's headline table.** The same per (method, saving), with the truth beside it |
| `TABLE_PLCAOracleWeights.csv.gz` | NB3 | the nine-method truth run: the six, plus three fitted under weights that know the true mode-level share |
| `TABLE_PLCAOracleSummary.csv` | NB3 | that, per family and weighting. **Read the framing note in the notebook before quoting it**, and note the 2026-09-29 correction: the contrast is NOT knowing shares against guessing them. The variable arm already carries the true group-level market share exactly; the oracle differs only in dividing a group's share evenly rather than at random, which decision 79 records as uninformative by construction. What ignoring a KNOWN market share costs is uniform against variable, and needs nothing from this table |
| `TABLE_PLCAFlipDrivers.csv` | NB3 | whether the top-two ratio decides a flip on its own. It nearly does |
| `TABLE_PLCASafeLead.csv` | NB3 | the lead a material needs, as a function of how spread the two materials are |
| `TABLE_PLCAFlipByLeadAndSpread.csv` | NB3 | **the table to print for that question.** The risk at a given lead, split by how many standard deviations the lead is worth, with counts beside every cell |
| `TABLE_PLCASeparationCeiling.csv` | NB3 | why the rule cannot be written in standard deviations: the measure saturates at 1 over the leading material's coefficient of variation |
| `TABLE_PLCABias.csv` | NB3 | each method's bias per material, its noise per material, and the systematic error that bias implies for the whole building |
| `TABLE_PLCANRMSE.csv` | NB3 | every pLCA output's NRMSE between the six methods, with a bootstrap interval. None had one before |
| `TABLE_ReductionSurvivors.csv` | NB3 | **the Stage 2f table to read.** Each candidate's mean permutation-importance rank pooled over every model that predicted anything, by target family, and with the definitional candidate removed |
| `TABLE_ReductionImportance.csv` | NB3 | one row per (arm, method, target, model, metric): the permutation importance, its spread across folds, and the model's own out-of-sample R2 |
| `AUDIT_ReductionIncremental.csv` | `audits/metric_reduction.py` | what each characteristic adds to predicting a target once a spline in log(n) is already in the model |
| `AUDIT_ReductionFitVersusAnswer.csv` | `audits/metric_reduction.py` | **the table Stage 2f's design exists for.** Each characteristic's rank against the FIT score beside its rank against the DOWNSTREAM error, and the shift |
| `TABLE_ReductionDefinitional.csv` | NB3 | whether a candidate PREDICTS a fit score or IS part of one. `w_v_uw_wasserstein` reproduces the definitional term exactly for every uniform-weighted method |
| `TABLE_ReductionSizeConfounding.csv` | NB3 | how much of each characteristic log(n) alone explains, by a spline fit |
| `TABLE_ReductionRedundancy.csv` | NB3 | the correlation structure and the effective dimension of the candidate set |
| `TABLE_ReductionMissingness.csv` | NB3 | how many datasets each characteristic is DEFINED on, per size band |
| `TABLE_ReductionRowsUsed.csv` | NB3 | how many datasets each MODEL uses, per size band. Catches the exclusion the missingness table cannot see: the cross-validated empirical target is undefined below n = 10 |
| `TABLE_ReductionCompleteCaseCost.csv` | NB3 | what a model that dropped incomplete rows would have thrown away |
| `TABLE_ReductionPostStratified.csv` | NB3 | every importance at equal allocation and on a corpus resampled to the empirical size mix |
| `AUDIT_ReductionWinner.csv` | `audits/metric_reduction.py` | whether WHICH METHOD WINS can be predicted, with the majority-class baseline beside every accuracy |
| `TABLE_ReductionModality.csv` | NB3 | the three modality measures head to head, each offered alone over a spline in log(n) |
| `TABLE_ReductionModalityAgreement.csv` | NB3 | how far apart the modality measures are, and the share each calls unimodal over the datasets it is DEFINED on |
| `AUDIT_ReductionPartialDependence.csv` | `audits/metric_reduction.py` | what each survivor is worth with the others HELD, as a curve |
| `AUDIT_ReductionMarginalVersusPartial.csv` | `audits/metric_reduction.py` | the marginal range beside the partial one, BOTH IN LOG UNITS of the target, and the ratio |
| `TABLE_ReductionCurves.csv.gz` | NB3 | the curves that replace the rolling averages: equal-count bins with a bootstrap band and a count, plus a LOWESS smooth |
| `TABLE_ReductionCoverageVsImportance.csv` | NB3 | **the generalization question, as a join.** Each candidate's importance beside how far the corpus reaches past the empirical range on it |
| `TABLE_MetricRecovery.csv` | NB3 | **the Stage 2g table to read.** Per (truth parent, candidate metric, method): the mean absolute error against the true parent with an interval, and TWO normalizations of it that answer different questions. `recovery` divides by `truth_sd`, the spread of the TRUE value across materials, and says whether the metric can tell two materials apart -- this is what ranks the CANDIDATE METRICS. `rel_error` divides by `truth_mean`, the LEVEL, and says how wrong the number is -- this is what compares one CLAIM with another and is what the scorecard draws. The level runs from 1.17 to 6.57 times the spread across the seven outputs, so the two are not interchangeable and must not share an axis |
| `TABLE_MetricDecisionAgreement.csv` | NB3 | the same question as an argmax: how often the method names the material the truth names, against chance |
| `TABLE_MetricVerdict.csv` | NB3 | **the join.** Recovery, agreement and NRMSE per metric in one row, which is what makes a metric every method agrees on and every method gets wrong visible |
| `TABLE_MetricWinShare.csv` | NB3 | how often each method is closest to the truth, for every candidate metric rather than the three an earlier stage picked |
| `TABLE_MetricTailStress.csv` | NB3 | what a thin far tail costs in W1, with the tail term and without it, beside what it costs in every output |
| `TABLE_MetricTailExposure.csv` | NB3 | the ratio: how far a metric moves per unit the goodness-of-fit criterion moves |
| `TABLE_CapReductionNormalization.csv` | NB3 | the two sums of the corrected cap rank frequencies, which are now both quantities |
| `TABLE_CapReductionByMethod.csv` | NB3 | how often the cap binds under each method, which the old constant divisor forced to 0.25 |
| `TABLE_FiveStatements.csv` | NB3 | **the results section in order.** The five statements a pLCA makes, each with the truth and the span across the six methods, assembled from the tables already on disk |
| `TABLE_MetricClaimScorecard.csv` | NB3 | **the claim-by-method table, and the one to print.** FIFTEEN claims grouped under the FIVE QUESTIONS a reader of a probabilistic LCA asks, each scored for all six methods against the truth on **one definition for every row**: `total_error` is the mean absolute error divided by the mean TRUE LEVEL of the same quantity, and it is what the figure shows in every cell. `best_error` is what the closest of the six still gets wrong, `excess` is each method's excess over it, and `stakes` is worst minus best. **`stakes` IS NOT WHAT THE CHOICE OF METHOD COSTS AND IS NOT THE RIGHT-HAND BAR**, which this row said until 2026-10-05. It is `max(mean error) - min(mean error)`, the spread of the AVERAGE error across methods, so it is an averaged quantity where every CELL of the same figure is the error in ONE decision (decision 174). The two part company wherever the methods are each about equally wrong but wrong about DIFFERENT buildings: on the uncertainty index `stakes` is 2.2 percent of the true level while the mean per-pair per-decision difference is 27.3 and the study's own NRMSE of 0.5504 for that output implies 44.0. **What the choice costs per decision is `pair_mean` in `TABLE_ClaimChoiceCost.csv`, and that is what the bar draws.** `best_error` and `stakes` still answer different questions from each other: a small spread can mean every method is right or every method is wrong. **An earlier version used two denominators** -- the between-material spread for the attribution rows and the true level for the rest -- and drew both on one color scale; those are a signal-to-noise ratio and a relative error, they are not a fixed multiple of each other, and decision 157 is the correction. `total_w1` was dropped with that change, because a distance has a true value of zero and no level to be a percentage of |
| `TABLE_ClaimChoiceCost.csv` | NB3 | **what the choice of method costs IN ONE DECISION**, which `stakes` is not. One row per claim, every column divided by the claim's mean true level so it is comparable with `total_error`. `pair_mean` is the mean over method PAIRS of the mean absolute difference PER UNIT and is the scorecard figure's right-hand bar; `pair_worst` is the same for the worst pair. `shared` is the part of the error every method makes, `specific` is `worst_cell - shared` so the two close on the worst method's own cell, and `stakes_mean` reproduces `rescore`'s `stakes` from the row-level frames as the control that this is the same experiment. **`pair_mean` and `specific` are not the same quantity and must not be quoted for each other**: on the uncertainty index they read 27.3 and 5.2 percent of the true level |
| `TABLE_MetricSizeBands.csv` | NB3 | the seven per-material claims by (dataset size band, method), each error as a pct of that claim's true level over the whole arm. **The scorecard's own caveat**: the pooled box count is an average over a size mix that is a design choice, and the ordering inverts at about 100 declarations. It is the figure's second panel |
| `TABLE_MetricConclusions.csv` | NB3 | whether the paper's existing claims survive the companion metrics: the method ordering under each, how much worse the normal is, and how far the four non-normal methods span |
| `TABLE_MetricWinLeaders.csv` | NB3 | whether each metric's win-share leader is separated from the runner-up or tied with it. On two of seven it is tied |
| `TABLE_CapApplicabilityVsTruth.csv` | NB3 | how often each method finds the specification cap binding, against how often it really does. Only the two lognormals are indistinguishable from the truth |
| `TABLE_MetricTailReality.csv` | NB3 | whether any model this study actually FITS has the runaway tail the stress test simulates. A quarter of a percent do, all of them equal-weighted fits to small datasets |
| `TABLE_MixedPolicyComposition.csv` | NB3 | per pLCA group: how many of its four materials the size rule puts on each side of the threshold, the smallest and largest dataset in it, and whether it straddles |
| `TABLE_MixedPolicyFit.csv` | NB3 | the seven policies on the FIT: mean and median W1 to the market parent, cost over the per-dataset oracle, the worst single dataset, and the head-to-head with the ties the rule creates by construction reported separately |
| `TABLE_MixedPolicyTruth.csv.gz` | NB3 | one row per (group, material, policy): every output, the true value, the error. 70,000 rows |
| `TABLE_MixedPolicyBuilding.csv.gz` | NB3 | one row per (group, policy): the building total against the truth |
| `TABLE_MixedPolicyIntervention.csv.gz` | NB3 | one row per (group, material, policy): what a cap and a quantity reduction deliver, against the truth |
| `TABLE_MixedPolicySwap.csv.gz` | NB3 | one row per (design pair, claimed saving, policy): P(B beats A) against the truth |
| `TABLE_MixedPolicyScorecard.csv` | NB3 | **the Stage 2j table to read beside the six-method one.** The fifteen claims by every policy the cutoff sweep builds -- 62 of them, the six fixed methods plus each swept cutoff and each one-axis variant -- both numerators, on the same divisor as `TABLE_MetricClaimScorecard.csv`. The six-method table is NOT superseded: `best_method`, `stakes` and `excess` there are properties of the six-policy set and stay the paper's comparison of METHODS. **`stakes` carries the same caveat here as above**: it is the spread of the AVERAGE error across policies, not what choosing one over another costs in one decision |
| `TABLE_MixedPolicyGain.csv` | NB3 | per claim: the size rule against the best FIXED policy on that claim, with a paired cluster-bootstrap interval. Positive means the rule is closer to the truth |
| `TABLE_MixedPolicyCeiling.csv` | NB3 | per claim: the best fixed policy, the unreachable per-material oracle, the rule, and the fraction of the distance it closes. **The oracle is a bound and never a policy**, and it is optimistic because a minimum over six noisy errors is biased low |
| `TABLE_MixedPolicyThreshold.csv` | NB3 | **the range to print.** One row per (rule family, cutoff): pooled error over all fifteen claims at each of fourteen cutoffs from 3 to 10,000, with a PAIRED penalty interval against whichever cutoff won on the same resample, and the flag for the longest unbroken run of cutoffs that cannot be told apart from the best |
| `TABLE_MixedPolicyRanking.csv` | NB3 | every policy and every fixed method on ONE number, pooled over the fifteen claims (`n_claims` carries it). This is where the four one-axis variants say that the WEIGHTING switch does the work and the family switch does not |
| `TABLE_MixedPolicyWeighting.csv` | NB3 | the share of datasets on which a fit using the TRUE market shares beats its own uniform-weighted twin, by size band, with the Kish effective sample size beside it. It crosses half at the cutoff, for both families, with nothing tuned to make it |
| `TABLE_MixedPolicyPooled.csv` | NB3 | pooled relative error over every claim belonging to a pLCA group, split by how many of the group's four materials the rule moves. **BOTH RULES since Stage 4** -- the known-share rule and the feasible one, each beside the two fixed methods it collapses to -- and each carries its own control: at 0 and at 4 a rule IS a fixed policy and must equal it exactly |

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
| `test_customstats.py` | 30 | hand-computed quantiles, order invariance with and without ties, moments against scipy, both bandwidth formulas, Wasserstein identities |
| `test_notebooks.py` | 31 | every code cell parses, no global numpy randomness, exactly one Generator per notebook, no cell reads a frame a later cell defines, no variable shadows an imported module, and notebook 3 writes only through its redirectable output root |
| `test_components.py` | 48 | moment targets hit exactly, infeasible targets refused not approximated, every accepted component inverts its own CDF, the four families partition the Pearson plane |
| `test_mixture.py` | 9 | the parent CDF matches a 400,000-draw sample, the market-weighted parent is a real population object, coupling 0 collapses the two parents, inverse-CDF sampling agrees with the truncation loop it replaced, overlap is symmetric and monotone in separation |
| `test_modality.py` | 8 | binned KDE matches direct evaluation, mode count ignores FFT round-off and is non-increasing in bandwidth, Silverman recovers known mode counts, the statistic is scale free and defined at n = 3 |
| `test_comparison.py` | 10 | held-out W1 is undefined below n = 10 rather than computed from two points, is worse than in-sample for the flexible method, and removes most of W1's bandwidth sensitivity without replacing it with a sharp optimum; the model-spread ratio catches a tail W1 does not; ranks are within-dataset and invariant to rescaling a dataset; the curve window scales to the arm instead of assuming the corpus; all six methods share each held-out split, so the comparison is paired |
| `test_recovery.py` | 26 | the parent spec round-trips exactly and the overlap displacements are NOT in the generation record, which is why the replay exists; a recovery score is zero when the model IS the parent and rises as it moves away; the grid always covers the parent; the two weightings are scored against different parents; the tail charge catches a far tail the body score does not; cross-validation is undefined below n = 10, penalizes the flexible method relative to in sample, and is paired across methods; the decomposition satisfies its own inequality and the definitional term is identical across uniform methods and zero for variable ones; regret is zero for the winner; post-stratification moves an aggregate toward the common band and the empirical shares are measured not assumed; a win share only moves when the WINNER moves, which is why the empirical headline is stated as one; the paired bootstrap finds a real gap and not an imaginary one |
| `test_weighting.py` | 13 | the location term is a lower bound on W1 and a two-point dataset is all location; every relative measure is invariant to rescaling the data while the absolute W1 is not, which is the control; A_IQR is exactly scale invariant, tracks sample size rather than dispersion while the mean-relative separation does the reverse -- with `A_IQR * sqrt(n)` and `separation / coeffvar` each pinned as the nearly-constant quantity, so the MECHANISM is asserted and not just the outcome -- falls with dataset size, is zero for a degenerate ensemble, and its component curves are densities |
| `test_flip.py` | 15 | common random numbers make a method identical to itself while independent streams do not, which is the control; the tempering control at t = 0 gives zero separation and no flip, and separation grows with the level; the logistic recovers a known curve and its crossing inverts its own fit; the isotonic fit is monotone, preserves the mean and drops no point; the CLUSTER bootstrap is more than twice as wide as a row bootstrap, which is why the resampling unit is the pLCA; a model's distance to itself is zero and a relative distance is scale invariant |
| `test_materialclass.py` | 7 | the tiers are a pure function of the category NAME and the whole assignment runs on a frame with no value column, so a tier cannot have been chosen because a method won on it; concrete, steel and every insulation variant land where a building-LCA reader expects |
| `test_families.py` | 105 | the support is open at zero and no sampler can emit an inadmissible value, cdf inverts ppf on every family, inverse-CDF sampling reproduces the model CDF, `rvs_from_uniform` is the same map `rvs` uses, truncation renormalizes rather than discarding mass, the weighted KDE matches gaussian_kde's density and integrates to its own CDF, the closed-form lognormal and gamma estimators beat their neighbors on the likelihood, the profile threshold stays strictly below min(x) and reaches the normal limit when the data asks for it, an unguarded joint fit walks into the pathology and the guarded one does not, the W1-optimal fit never scores worse than the MLE fit |
| `test_plca.py` | 55 | common random numbers make a method identical to itself while independent variates do not, and sharing them leaves each method's own marginal distribution alone, which is what makes installing them a refinement rather than a change of estimand; materials stay independent within an iteration; the outputs are the notebook's own definitions, checked against its pandas ranking and against NRMSE computed the way the plotting function computes it; an infinite Dirichlet concentration reproduces the equal-intensity case EXACTLY and every intensity vector averages to 1.0; concentration makes the top contributor stop moving; resampled groups hold distinct datasets; the cluster bootstrap is more than twice as wide as a row bootstrap; the tabulated parent sampler inverts the parent's own bisection and stays inside its support; a method that IS the parent has exactly zero error, which is the truth run's control; the lazy samplers agree with eager ones while bounding their memory; and the committed pLCA table is a full run rather than a smoke one |
| `test_generator.py` | 18 | strata allocate and cover their endpoints, the probe set sits outside the corpus, generated datasets are valid and normalized, the record reconstructs the parent, the validity filter passes extreme-but-analysable data and catches unanalysable data, undefined kurtosis at n = 3 is not a failure, generation is reproducible and never touches global numpy state |
| `test_metricreduction.py` | 59 | a cross-validated gain cannot be bought by adding a useless term and its fold spread grows as the data thin; a negative R2 is reported rather than clipped, which is what exposed the empirical arm; forward selection refuses a near-duplicate column; the policy curve puts its flat region around the true crossover and beats both fixed policies; the effective sample size matches its closed forms; a transform propagates an undefined metric instead of inventing a value; every candidate has a declared modeling scale; the missingness report names kurtosis and the complete-case cost names the band it would drop, while both models still report the FULL row count; the reduction recovers a planted signal and ranks noise below it, and finds nothing when there is nothing, which is the control; an importance from a model that predicts nothing is refused a rank; size confounding catches a metric that IS log(n) in disguise; a bootstrap band widens where the data thin out and equal-count bins hold equal counts; the winner model reports its majority baseline beside its accuracy; partial dependence separates a real effect from a borrowed one AND retains a near-copy, which is the caveat the docstring records; the marginal and partial ranges are both in log units; log(n) comes from the frame and not from the candidate list; the unimodal share uses the denominator the measure is defined on |
| `test_metricset.py` | 28 | the new companion is a SHARE read at the BUILDING's bad end and not at the material's, with a planted case where one material drives the total's upper tail and the two metrics have to disagree; the corrected cap rank-1 frequencies sum to exactly 1.0 across the materials and are the old count divided by the measured applicability; a method that IS the truth has exactly zero recovery error, recovery grows with the error, and a method that reports one number for every material cannot score better than one that tracks the truth with noise; the argmax agreement is 1.0 for the truth and chance for a shuffle; the contaminated model inverts its own mixture CDF and reduces to its base at zero weight; **W1 over the scoring grid alone gives the SAME score at ten, a hundred and a thousand times the mean while the tail term rises with the distance**; and a share saturates under contamination while a mean, a standard deviation and a variance share do not. **Two tests pin the two divisors apart**: `rel_error` is the error over the true LEVEL and `recovery` the error over the true SPREAD, their ratio is not one, and a single planted near-zero truth cannot move `rel_error`, which is why it is a ratio of means and not the ordinary mean absolute percentage error. **Three more pin this stage's review**: the display rename maps only the weighting and leaves `fitting.PEWT` untouched; the size-band split recovers an ordering that flips with dataset size where the pooled table cannot see it; and an output with a true value of zero is skipped rather than divided by |
| `test_mixedpolicy.py` | 36 | the rule reads NOTHING but the dataset's size, and the interface cannot express a second selector; adding the policy refits nothing and the selected model object is the same object, so a group entirely on one side of the threshold reproduces that fixed policy bit for bit and the six fixed policies come back identical from a truth run whether or not the seventh is present; the fit score is a SELECTION out of the six-method table and cannot disagree with it; the paired bootstrap agrees with `plca.cluster_bootstrap`; a policy that IS a fixed policy on every unit scores a gain of exactly zero with an interval that closes on it, which is the null control; the oracle is a per-unit minimum and not a grand mean, which is the defect the first draft had; a design PAIR is not given a pLCA group's composition even though their ids collide; and no display label uses the retired weighting vocabulary. **The sweep adds**: every swept cutoff and every variant reads only the size; the study's cutoff keeps its bare name inside the sweep so older tables still join; the gain's comparator is one of the SIX FIXED methods and never a neighboring cutoff, which would collapse it; a planted curve's minimum and a range around it are recovered, and a flat curve gives a wider range than a steep one; and pooling covers the design comparison's own unit universe rather than leaving it a column of NaN. **And the two rule families**: the feasible one uses uniform weights on both sides, a curve over both families at once is refused, and the synthetic market weights carry the TRUE group-level shares, which is the claim three stages described backwards |
| `test_remetric.py` | 3 | `remetric_corpus` relabels the parent-spec replay cache it copies, a cache from a genuinely DIFFERENT corpus is still refused, and the values are copied byte for byte while the characteristics really are recomputed |

`test_notebooks.py::test_all_code_cells_parse` exists because a Stage 1 patch
script silently dropped the final line of any cell whose source did not end in
a newline, truncating a cell mid-statement. The only symptom was a SyntaxError
twelve minutes into a headless run.


## Four audits added by Stage 2h, and two of them are GATES

**`audits/parent_sampler_fidelity.py` and `audits/draft_end_to_end.py` must
both pass before any generator configuration supplies a corpus.** They exist
because a regeneration once passed every sample-level check, IMPROVED the
calibration objective, and produced a run against the true parents with 99.98
percent errors. Nothing in that stage looked at the object the truth run
actually draws from, which is neither the generator nor the sample.

`parent_sampler_fidelity.py` compares `plca.ParentSampler`'s interpolated
inverse CDF against the parent's own bisection at thirteen probabilities from
1e-6 to 1 - 1e-6, over four dataset sizes and both weighting schemes, then
draws 10,000 values and checks the realized mean against an exact mean computed
on a DIFFERENT node set. The shipped configuration and every bounded candidate
score 5e-4; the configuration Stage 2h rejected is wrong by more than 1 percent
on 55 percent of its parents. `--set K=V` overrides a parameter, repeatable.
About 100 seconds at 30 parents per cell.

`draft_end_to_end.py` runs the whole pipeline on a candidate corpus -- replay
the parents, fit all six methods, score each against its recovered truth --
and range-checks three numbers. **Its band is MEASURED, not recalled**: run it
on the corpus already in the paper to calibrate, which is what the first
version of it got wrong. It loads cached parent specs rather than replaying
when they exist, and it samples stratified across the size range rather than
alphabetically, because an alphabetical slice lands almost entirely in the
smallest size band.

    python audits/parent_sampler_fidelity.py --parents 40 --label "candidate"
    python audits/draft_end_to_end.py <corpus label> 300

`audits/widening_and_weights.py` scores ONE synthetic draw against the
empirical arm built under four different weight rules, so the columns differ
only in how the real categories were weighted. It is what showed that the
dispersion-versus-weighting trade three stages called structural was an
artifact of the two arms weighting differently. Decision 193.

`audits/pedigree_range.py` enumerates all 3,125 pedigree score combinations
from Muller et al. (2016) and reports what geometric standard deviation the
matrix can produce, against the real categories' own. **Every factor in that
table contributes to the SQUARE of the geometric standard deviation**, so a
model quoted as a GSD halves the exponent; getting that wrong doubles the
spread. Decision 194.

## Two audits added by the Stage 2g review

`audits/lognormal_variants.py` scores the two-parameter lognormal, this study's
three-parameter one, gamma, the normal and the kernel estimate against the KNOWN
PARENT under uniform weights, and reports each one's gain over the two-parameter
fit. It exists because the field's lognormal is the two-parameter one and the
study's is not, and a reader who assumes they are the same will read the
scorecard as "the paper rediscovered current practice". Writes
`outputs/tables/audits/TABLE_LognormalVariants.csv`. About 20 seconds at 1,500
datasets.

`audits/corpus_modality_shape.py` correlates the visible mode count with every
other characteristic, separately on each arm, and reports the generator-side
mechanism and the size of the gap. It exists because decision 82 closed the
modality question by REWEIGHTING the corpus to the empirical mode mix, and
reweighting cannot create a population the corpus does not contain. Writes
`TABLE_CorpusModalityShape.csv` and `TABLE_CorpusModalityHole.csv` in the same
directory. Seconds.

Both write only under `outputs/tables/audits/`, which is what an audit script is
permitted to touch.


## Judging a candidate generator configuration

Three instruments, fastest first, added by the Stage 2g review when the author
asked whether the corpus under-represents multimodal datasets.

`audits/shoulder_probe.py` draws parents directly and samples them: about 30
seconds a variant. Use it to SEARCH. It reports the share of datasets that are
multimodal, the share dispersed, the share that are both, the conditional
share, and the sign of the modality-shape correlations, against the real arm's
targets.

`audits/corpus_joint_structure.py` builds a real DRAFT CORPUS per candidate on
the stratified design, about 100 seconds each, and adds the marginal
calibration objective and the two margins that matter most -- dispersion and
`w_v_uw_wasserstein`. Use it to JUDGE. It reuses an existing draft rather than
relabeling, because a corpus is never overwritten.

`audits/corpus_examples.py` draws notebook 1's example-dataset panels for any
candidate, real categories in the bottom row. Use it to LOOK. The probe numbers
say what changed; only this says whether the datasets look like ECC data.

**The probe and the corpus disagree on levels and agree on ranking**, because
the probe samples dataset size log-uniformly and the corpus samples it by
stratum. Rank candidates on the probe; quote numbers from the corpus.

**Both generator options they exercise default to OFF and are covered by
`tests/test_determinism.py`**: `genconfig.shoulder_frac` with `shoulder_body`,
and `genconfig.separation_dispersion_frac`. The second is the one that works
and is not adopted; decision 170 says why.


## Redrawing a figure without re-running its notebook

`audits/render_figures.py`, 2026-09-22. Notebook 4 takes about twenty minutes
end to end and almost all of it is model fitting; a figure iteration changes
twenty lines that read tables already on disk.

    python audits/render_figures.py 04_CompareUQ_ReduceMetrics --out /tmp/figs
    python audits/render_figures.py 04_CompareUQ_ReduceMetrics --into-outputs
    python audits/render_figures.py 04_CompareUQ_ReduceMetrics --only "FIGURE B"

Seven seconds against twenty minutes. **The script holds no figure code**: it
executes the notebook's own setup cell and then its own figure cells, verbatim,
so there is nothing here to drift out of step with the notebook.
`tests/test_render_figures.py` asserts the executed source is byte-identical to
the notebook's and that the module contains no plotting call of its own, which
is what narrows decision 56 safely.

Without `--into-outputs` it writes to a scratch directory whose `tables/` is a
symlink to the real one, so the figures read production data while `outputs/`
is untouched. It REFUSES a notebook with an unmarked figure cell rather than
redrawing only some of them; a figure cell announces itself by starting with
`# FIGURE` or `# SUPPLEMENT`.

**ALL FOUR NOTEBOOKS ARE ACCEPTED AS OF STAGE 3.** Notebooks 1 and 2 had TWO
blockers and the first is the one that is easy to miss: the tool raises on a
missing `OUT` BEFORE it checks markers, so marking alone unlocked nothing.
Both now define `OUT` in their configuration cell and every figure cell reads
its data from a table.

**BEING ACCEPTED IS NOT THE SAME AS EVERY CELL BEING RENDERABLE, and `--only`
hides the difference.** The tool runs the SETUP BLOCK -- every code cell up to
and including the one defining `OUT` -- and then a figure cell; a cell that
reads a frame or a helper defined in a compute cell in between will raise.
Render one self-sufficient cell with `--only` and the tool reports success.
Stage 3 cleared notebooks 1, 2 and 4 that way and left notebook 3 at 28 of 37,
which was believed clear for exactly that reason.

**ALL 37 OF 37 PASS FROM STAGE 4**, and clearing notebook 3's nine took four
kinds of change, all of them the compute/plot split Stage 3 did elsewhere:

  - the display constants moved INTO the setup cell -- `PE`, `WT`,
    `dct_colors` and `dct_resultlabels`, the last of which seven figure cells
    read -- alongside two loaders, `figure_plca()` and `figure_combos()`, and
    `result_columns()`, which derives the output list the figure panels are
    laid out in. The cell that built `df_stds` asserts its own list against
    that function, so a figure drawn from disk cannot reorder its panels
    against one drawn from kernel state;
  - `compare_results` and `compare_results_bypewt` moved into the setup cell
    and take their two frames as ARGUMENTS rather than reading globals, which
    is what made that possible;
  - four frames that existed only in memory are persisted:
    `TABLE_PLCAInterMethodDistance.parquet` and the four
    `TABLE_PLCAExample*` tables, the last of which carry the three case-study
    pLCAs' values, fitted densities, W1 matrices and whole-building draws;
  - **a figure cell was consuming the notebook's random stream.** The
    whole-building draws in `FIG_PLCAVisualizeUQFits` were taken with
    `rvs(..., random_state=rng)` INSIDE the figure cell, so redrawing the
    figure moved every number after it. They are taken in the compute cell
    now, in the same place in the stream, and the figure reads them.

`python audits/figure_manifest.py` writes
`outputs/tables/audits/TABLE_FigureRendererSafety.csv` and the suite fails if
the count regresses.

### Writing a figure

`figstyle.savefig(fig, OUT, stem)` and nothing else. It writes
`CompareUQMethods_<stem>.png` plus a `.pdf` sibling from one call, so the
raster and the vector cannot drift apart, and it RAISES if two places in one
process write the same stem. This repository carried two different figures both
called `FIG2`; a duplicate filename fails silently and the second writer wins.

`python audits/figure_manifest.py` joins every image on disk against the code
that writes it and reports orphans, duplicates, style-module compliance and
renderer safety. `tests/test_figure_manifest.py` fails on an orphan, a
duplicate, a retired weighting word in a filename, or a PNG with no vector
sibling.
