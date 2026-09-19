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
├── FIGURE_STYLE.md            how every figure is built, after Tufte and
│                              Doumont. Read before writing a figure
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
│   ├── comparison.py          the paper's method comparison: held-out W1, the
│   │                          tail check, ranks and characteristic curves
│   ├── materialclass.py       structural / envelope / other, from the category
│   │                          NAME only, so the comparison can be read by what
│   │                          a material IS (Stage 2c)
│   ├── figstyle.py            FIGURE_STYLE.md in code: palette, rcParams,
│   │                          direct labelling, the greyscale check
│   ├── reduction.py           which characteristics carry signal (Stage 2f):
│   │                          the candidate set and its transforms, explicit
│   │                          missingness, the size confound, two model
│   │                          families ranked by permutation importance, the
│   │                          tautology guard, and the binned and LOWESS
│   │                          curves with bootstrap bands that replace the
│   │                          rolling averages
│   ├── weighting.py           does the weighting scheme matter, per dataset
│   │                          (Stage 2d): the location/shape split of the
│   │                          uniform-to-variable W1, the named relative
│   │                          measure, and A_IQR from the KL2 paper
│   ├── flip.py                what a given W1 COSTS (Stage 2d): the
│   │                          common-random-numbers pLCA, model-to-model
│   │                          distances, and the calibration curve
│   ├── plca.py                the pLCA CONSTRUCTION (Stage 2e): common random
│   │                          numbers, the group-size and material-use-
│   │                          intensity sweep, the cluster bootstrap, NRMSE
│   │                          with an interval, and the run against the TRUE
│   │                          parents
│   ├── recovery.py            the evaluation target (Stage 2c): W1 against the
│   │                          known parent, cross-validation, the
│   │                          fit-versus-definitional split, regret,
│   │                          post-stratification, the paired bootstrap
│   ├── datavisualization.py   one color helper
│   ├── funcs_unit_conversion.py  EC3 unit normalization
│   └── dct_metriclabels.json  display labels for the 22 metrics
├── audits/                    one-off measurement scripts, each named for what
│                              it measures; see audits/README.md
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
largest contributor crosses 1, 5 and 10 percent at relative W1 of **0.0018,
0.011 and 0.025**, in units of the dataset's own unweighted mean.
`flip.FLIP_THRESHOLDS` carries them, rounded to two significant figures because
the bootstrap interval is about 30 percent wide and an independent run differed
in the third figure.

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

## 2c. Which characteristics carry signal

`src/reduction.py`, Stage 2f, called from the last section of notebook 3.

**IT IS IN NOTEBOOK 3 AND NOT NOTEBOOK 2 FOR A REASON.** The reduction is run
against two kinds of target: the FIT score, how far a fitted curve sits from
its target, and the ANSWER, how wrong the probabilistic LCA's output is against
the true parent. The second exists only after `plca.truth_run`, so putting the
reduction in notebook 2 would make notebook 2 read a table notebook 3 writes
and break the run order on a clean clone.

**23 candidates**: the 21 characteristics the old figure drew, plus the two
visible-mode counts, so that all three modality measures are separate
predictors. `METRIC_TRANSFORM` gives each one its modeling scale, because a
spline basis on a raw quantity spanning four orders of magnitude puts every
knot in the first percent of the range.

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
row; `lowess_curve` gives a smooth whose band comes from resampling the
datasets and a local count for the density rug. The old rolling-average figure
is kept beside the new one so the two can be checked against each other.

---

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
corpus_2026-09-15b all 10,050 datasets replay byte-identically, in 13 minutes.
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
python -m pytest tests/          # 471 tests, about 160 seconds
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

Approximate runtimes on a 2026 laptop: **NB1 about 30 min**, **NB2 about 35
min**, **NB3 about 2 h** at `neccs = 10000`. NB1's figure was recorded as
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
| `TABLE_MethodCurves.csv.gz` | NB2 | every score against every characteristic, unbinned, with the rolling mean the figures draw |
| `TABLE_BandwidthRules.csv` | NB2 | the two KDE methods under all three bandwidth rules |
| `TABLE_TargetComparison.csv` | NB2 | **the Stage 2c table to read.** One row per (arm, dataset, method): the in-sample score, the recovery score against the parent it estimates and against the market parent, the cross-validated score with its spread across splits, the decomposition and the overlap area |
| `TABLE_TargetSummary.csv` | NB2 | the six methods by arm, criterion and weighting scheme |
| `TABLE_CrossValidatedScores.csv.gz` | NB2 | one row per (dataset, method, split, direction) |
| `TABLE_CrossValidatedSummary.csv` | NB2 | the mean over splits and the spread across them |
| `TABLE_PairedBootstrap.csv` | NB2 | whether a gap between two methods survives resampling the datasets |
| `TABLE_WeightingDecomposition.csv` | NB2 | fit error against the definitional gap |
| `TABLE_WeightingOnCommonTarget.csv` | NB2 | does variable weighting help, on the market parent, by size band |
| `TABLE_Regret.csv` | NB2 | mean, median and upper tail of regret per method |
| `TABLE_PostStratifiedScores.csv` | NB2 | every headline aggregate equally allocated and reweighted. **NOT `TABLE_PostStratified.csv`, which is NB1's and is about the dataset characteristics** |
| `TABLE_ModalityConditioned.csv` | NB2 | the method comparison split by visible modality, within size band |
| `TABLE_PolicyComparison.csv` | NB2 | **the table a practitioner reads.** Each fixed and size-conditional policy against the per-dataset oracle: mean cost, share within 5 and 20 percent of the best, and the worst single dataset |
| `TABLE_RuleSelection.csv` | NB2 | whether adding a characteristic to the rule beats a size threshold alone. It does not |
| `TABLE_RuleCandidates.csv` | NB2 | how much each characteristic adds to predicting the KDE-lognormal gap once log(n) is in the model |
| `TABLE_SizeCrossover.csv` | NB2 | the fitted slope and break-even n for each arm |
| `TABLE_SizeVersusMaterial.csv` | NB2 | the same fit used by the material figure |
| `TABLE_MaterialTiers.csv` | NB2 | every category with its material tier and size. **Publish this**: a hot-spot argument cannot be checked without it |
| `TABLE_MethodByMaterialTier.csv` | NB2 | the method comparison inside each tier |
| `TABLE_CharacteristicsByTier.csv` | NB2 | median characteristics by tier, which is why the tiers differ |
| `TABLE_MaterialTiers.csv` | NB2 | every category with its material tier. **Publish this**: a hot-spot argument cannot be checked without it |
| `TABLE_MethodByMaterialTier.csv` | NB2 | the method comparison inside each tier, and for structural categories at n >= 100 |
| `TABLE_VisibleModes.csv` | NB1 | visible modes per dataset at scipy's default bandwidth and at the one the study fits |
| `TABLE_VisibleModeSummary.csv` | NB1 | the share with one, two, three or more visible modes, at both bandwidths |

**Figures added in Stage 2c:** `FIG_EvaluationTarget`, `FIG_TargetBySize`,
`FIG_Regret`, `FIG_MethodByMaterial` (which is the POLICY comparison, not a
material breakdown -- the tier is not a mechanism, decision 84) and
`SUPP_AllEmpiricalFits`, all 147 empirical datasets with all six fits.
| `TABLE_MethodWinShare.csv.gz` | NB2 | how often each method wins, against the percentile of each characteristic |
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
| `TABLE_PLCATruth.csv.gz` | NB3 | **the Stage 2e table to read.** One row per (pLCA, material, method, truth parent): every output, the value the TRUE parent gives, and the error |
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
| `TABLE_PLCAOracleSummary.csv` | NB3 | that, per family and weighting. **Read the framing note in the notebook before quoting it**: the contrast is knowing shares against guessing them, not uniform against variable |
| `TABLE_PLCAFlipDrivers.csv` | NB3 | whether the top-two ratio decides a flip on its own. It nearly does |
| `TABLE_PLCASafeLead.csv` | NB3 | the lead a material needs, as a function of how spread the two materials are |
| `TABLE_PLCAFlipByLeadAndSpread.csv` | NB3 | **the table to print for that question.** The risk at a given lead, split by how many standard deviations the lead is worth, with counts beside every cell |
| `TABLE_PLCASeparationCeiling.csv` | NB3 | why the rule cannot be written in standard deviations: the measure saturates at 1 over the leading material's coefficient of variation |
| `TABLE_PLCABias.csv` | NB3 | each method's bias per material, its noise per material, and the systematic error that bias implies for the whole building |
| `TABLE_PLCANRMSE.csv` | NB3 | every pLCA output's NRMSE between the six methods, with a bootstrap interval. None had one before |

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
| `test_plca.py` | 54 | common random numbers make a method identical to itself while independent variates do not, and sharing them leaves each method's own marginal distribution alone, which is what makes installing them a refinement rather than a change of estimand; materials stay independent within an iteration; the outputs are the notebook's own definitions, checked against its pandas ranking and against NRMSE computed the way the plotting function computes it; an infinite Dirichlet concentration reproduces the equal-intensity case EXACTLY and every intensity vector averages to 1.0; concentration makes the top contributor stop moving; resampled groups hold distinct datasets; the cluster bootstrap is more than twice as wide as a row bootstrap; the tabulated parent sampler inverts the parent's own bisection and stays inside its support; a method that IS the parent has exactly zero error, which is the truth run's control; the lazy samplers agree with eager ones while bounding their memory; and the committed pLCA table is a full run rather than a smoke one |
| `test_generator.py` | 18 | strata allocate and cover their endpoints, the probe set sits outside the corpus, generated datasets are valid and normalized, the record reconstructs the parent, the validity filter passes extreme-but-analysable data and catches unanalysable data, undefined kurtosis at n = 3 is not a failure, generation is reproducible and never touches global numpy state |

`test_notebooks.py::test_all_code_cells_parse` exists because a Stage 1 patch
script silently dropped the final line of any cell whose source did not end in
a newline, truncating a cell mid-statement. The only symptom was a SyntaxError
twelve minutes into a headless run.
