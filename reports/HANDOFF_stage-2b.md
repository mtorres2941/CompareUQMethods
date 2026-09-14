# HANDOFF stage-2b - The lognormal, the support, and fitting by the criterion

## 0. STATUS, read this first

**THE PIPELINE RUNS END TO END FOR THE FIRST TIME.** Notebooks 1, 2 and 3 all
execute clean against `corpus_2026-09-14d` with zero errors. That was the oldest
outstanding item in the project and it is closed.

**FOUR THINGS IN THIS STAGE COULD CHANGE WHAT THE PAPER CONCLUDES.** They are
listed here rather than left to be found in section 4.

| | finding | where |
|---|---|---|
| 1 | **The KDE is not the best method on the empirical arm, and was not before this stage either.** Under maximum likelihood three of five parametric families beat it; fitted by W1, all five do. On the synthetic corpus the properly fitted lognormal now beats it on mean W1 too, and the KDE's lead is down to the median. The paper cannot say "KDE fits best" without saying on which arm and by which summary | 4.5, entries 40 and 42 |
| 2 | **The empirical W1 values the manuscript reports are in the raw unit of each category.** Stage 1 scored the empirical arm on UNNORMALIZED values while reporting normalized metrics beside them. Mean 9.38 against 0.48 now. The RANKS are unaffected; the VALUES cannot be converted and must come from the rerun | 4.6, entry 41 |
| 3 | **The +0.5 offset was patching near-zero values, not the threshold pathology**, and the two-parameter lognormal is not the simple replacement: it is the worst of the three lognormals. The profile-likelihood three-parameter fit is what stands | 4.4, entries 37 and 40 |
| 4 | **The plausibility ceiling moved the tuning objective by 1.10 seed-to-seed standard deviations, marginally outside the gate.** Nothing was regenerated, which is the instruction. Section 4.2 says why 1.10 sd overstates it and what the author has to decide, if anything | 4.2 |
| 5 | **A fitted model can score well on W1 and still be unusable in the pLCA.** The first version of the profile-likelihood lognormal produced models with standard deviations in the thousands on data whose own standard deviation is 0.6, because W1 barely charges for a thin far tail. It was found in the pLCA results, not in the fit scores. Fixed by widening the threshold guard, which also IMPROVED W1 | 4.9 |

**Two things that moved almost nothing, and that is the finding.** Truncating the
normal and the KDE to (0, inf) explicitly changes the scored values by exactly
zero, because the Stage 1 grid started at zero and was already truncating and
renormalizing without saying so. Opening the grid at zero moves them by 0.1 to
0.4 percent of the median. What was wrong was the manuscript's DESCRIPTION, not
the arithmetic.

## 1. Stage and branch

- Stage: 2b, the lognormal and fitting by the scoring criterion
- Branch: `stage-2b-fitting`
- Branched from: `06dd270` "Make handoff-only communication a standing
  constraint", on `stage-2a3-categories`. The working tree was clean, so nothing
  needed committing first.

| Commit | Label |
|---|---|
| `aa446f5` Remove physically implausible mass-declared records | **MOVES NUMBERS** |
| `e004db4` Run all three notebooks against corpus_2026-09-14d, and re-freeze the fixtures | **MOVES NUMBERS** |
| `da27c1a` Fit the lognormal by profile likelihood, on the settled support (0, inf) | **MOVES NUMBERS** |
| this commit: the handoff | records only |

**All three notebooks were run to completion twice**, once at `e004db4` with the
Stage 1 fitting to isolate the corpus change, and once at `da27c1a` with the
Stage 2b fitting to isolate the method change. That is what makes any number
that moves bisectable to one decision. `e004db4`'s
`outputs/tables/TABLE_PLCAResults.csv` is the Stage 1 method against the new
corpus, and it is the comparison every pLCA figure in section 4.10 is against.

## 2. What was asked

Four tasks before the lognormal work: remove physically implausible records by an
external bound and report the extremes no bound can rule on; run notebooks 2 and
3 against the active corpus; re-freeze both metric fixtures; mark decision 13
confirmed. Then: determine what the code actually does with the lognormal
threshold, determine which of two problems the +0.5 offset was solving, implement
the profile-likelihood treatment of the unbounded three-parameter likelihood,
put every method on the support (0, inf) with inverse-CDF sampling, and implement
W1-optimal fitting alongside maximum likelihood for every parametric family.

## 3. What was done

### 3.1 The plausibility ceiling

`empirical.MASS_ECC_CEILING = 100.0` kgCO2e/kg, on mass-declared records only,
applied BEFORE cleaning so that a record wrong by three orders of magnitude is
not setting the interquartile range the cleaning rule is computed from.

**The bound is external and the reasoning is the point.** This study measures
dispersion and modality, so a ceiling read off the arm's own quantiles would be
circular in the way a dispersion-based split would have been (decision 46). It is
anchored on published embodied-carbon inventories, where the highest
building-product coefficients are of order 13 kgCO2e/kg for primary aluminium
(ICE v3.0), and cross-checked from a different direction with no database at all:
combusting pure carbon yields 3.67 kg CO2 per kg of carbon, so 100 kgCO2e per kg
of DELIVERED PRODUCT requires burning about 27 kg of pure carbon for every
kilogram shipped. Set at 100 rather than at 25 deliberately, so that it cannot be
read as a tuned threshold, and it still catches the known cases by two orders of
magnitude.

**VERIFY THE ICE FIGURE BEFORE IT GOES IN THE PAPER.** `refs/` holds no copy of
the ICE database and the 13 kgCO2e/kg is from the analyst's knowledge, not from a
document in this repository. The argument does not depend on the exact value --
anything in the 10 to 25 range leaves a ceiling of 100 generous -- but the paper
will cite a number and that number needs a source.

**Where no external bound exists, nothing was invented.** For volume, area,
length and item declarations the ten highest and ten lowest records per unit type
are reported with product names, declared units and their ratio to the category
median, for author review:
`outputs/tables/stage2b/TABLE_2b_UnitExtremes.csv`. It is a report, not a filter.
Worth the author's eye: `BlanketInsulation [mineral wool]` holds a ceramic fibre
blanket at 2,300 kgCO2e/m2, 1,564 times its category median, and
`ReadyMix [4000-4999 psi]` holds two mixes at 109,292 and 97,859 kgCO2e/m3
against a median near 335.

### 3.2 Notebooks 2 and 3, and two defects they exposed

Both defects are in the NOTEBOOKS, not in the corpus, so generation stayed closed.

**The pLCA grouping was dropping three datasets silently.** `make_combos`
truncated with an integer division. The corpus holds 9,999 datasets because one
parent failed to solve and was reported rather than approximated, so it no longer
divides by four. The remainder is still held out -- which is right, and is now
argued rather than implied, because every downstream rank metric is a rank among
exactly four materials and the headline result is a frequency over those ranks,
so one short group would put a rank-1 frequency of 1/3 in the same column as one
of 1/4. `corpus.describe_combos` names `dataset813`, `dataset2876` and
`dataset7985`, and both notebooks print the line. **2,499 pLCAs, not 2,500.**

**Notebook 3 cell 37 indexed a results frame by every dataset in the corpus**
while the results covered the 9,996 the grouping reaches, and cell 56 then fed a
9,999-row and a 9,996-row frame to `pearsonr`. It surfaced 18 minutes into a full
run. CONTEXT.md had recorded that those cells "assume every dataset appears in
some pLCA, which is only true in a full run"; that was no longer true of a full
run either, and both the code and the note are fixed.

### 3.3 What the code actually does with the lognormal threshold

The Methods text says in one place that shape, location and scale are all
estimated and in another that location is held at zero. **Neither is what runs.**

`customstats.weighted_lognorm_fit` returns `loc = 0.0` unconditionally -- its own
docstring says "Location parameter (always 0 in this fit)" -- and optimizes over
(sigma, mu) only. `fit_pewt_models` then builds
`lognorm(s, loc = 0 - LOGFIT_OFFSET, scale)`. The fitted threshold is a
**constant of -0.5, set by hand and never estimated**. The family in use is a
three-parameter lognormal with two free parameters and a hand-set threshold.

**A consequence that decides the next question.** Because the threshold is never
estimated, the unbounded-likelihood pathology CANNOT arise in the Stage 1 code:
there is no optimizer over the threshold for it to break. The pathology is real,
and it is a property of the three-parameter fit the Methods text CLAIMS, not of
the two-parameter fit the code performs. So the offset was not patching it.

**A second finding in the same function.** Its "MLE" branch hands
`scipy.optimize` a problem whose solution is closed form -- the weighted MLE of a
lognormal IS the weighted mean and standard deviation of `log x`, which is
exactly what its own "MoM" branch computes. On `Cement` the two agree to
0.000e+00. That is why the regression fixtures needed `rtol = 4.3e-08` on the
lognormal column while every other column agreed to 1.3e-14: the only
non-reproducible number in the fixture set came from an optimizer that was not
needed. Entry 38.

### 3.4 The new fitting machinery

`src/families.py` is new. `src/fitting.py` keeps the Stage 1 implementation
verbatim above a dividing line so the change from it can be measured rather than
asserted.

- `Truncated` wraps any frozen distribution, restricting it to (0, inf) and
  renormalizing, and exposes `pdf`, `cdf`, `ppf`, `rvs` and `rvs_from_uniform`.
- `WeightedKDE` replaces `scipy.stats.gaussian_kde`, which offers a density and
  a resampler and no CDF, so a KDE scored through it cannot be the object sampled
  from it. Its density matches gaussian_kde to machine precision; its CDF is the
  closed-form weighted sum of normal CDFs.
- `fit_lognorm3_profile` grids the threshold over a closed interval bounded
  strictly below `min(x)`, solves (mu, sigma) in closed form at each point, and
  takes the interior local maximum of the profile likelihood.
- `fit_lognorm2_mle`, `fit_gamma_mle`, `fit_normal_mle` are the closed-form or
  score-equation estimators.
- `fit_family(name, x, w, method='w1')` minimizes W1 directly, from the
  maximum-likelihood fit, so it can never score worse.
- `fit_pewt` is the production entry point and returns `(models, params)`, so a
  fit that hit a boundary is visible rather than assumed.

**Two implementation facts a later stage will need.** The KDE's kernel sums are
BLOCKED over query points: a 20,001-point lattice against the arm's largest
dataset would otherwise allocate a 20,001 x 31,025 array, about 5 GB. And the
KDE's `ppf` inverts an FFT-binned tabulation of its CDF rather than the CDF
itself, because direct inversion costs O(iterations x queries x n) and the pLCA
draws 10,000 values from each of about 20,000 fitted KDEs -- 11 seconds a call
against 2 milliseconds. `ppf_exact` is kept as the reference and
`tests/test_families.py` pins the two together: pushing a uniform through `ppf`
and back through the exact `cdf` returns it to within **1e-6 in probability**.

### 3.5 The scoring grid, stated

`fitting.score_grid_open`: **1,000 equally spaced points from `hi / 1000` to
`hi = max(x) + 10 * spread`**, where `spread` is the larger of the unweighted and
the weighted standard deviation. Open at zero.

**How the lower bound was chosen: it is the first point of the grid's own lattice
above zero, and it is chosen that way precisely so that it is not a choice.**
Decision 13 says an ECC of exactly zero is not admissible, so zero must leave the
grid. Any invented epsilon -- 1e-6, a fraction of `min(x)`, a quantile of the
fitted model -- would be a new free parameter that a reviewer could ask about and
that would have to be swept. The lattice already exists and already sets the
resolution of the whole calculation.

**What this does NOT fix, and Stage 2c owns it.** The lattice is LINEAR, so its
resolution near zero is `hi / 1000` for every dataset. On a dataset spanning
several orders of magnitude the interval below the first grid point contains real
data and everything the model puts there is lumped onto one point. That was true
before this change and is unchanged by it.

## 4. Numbers that moved

### 4.1 The plausibility ceiling

115 of 117,807 raw records (0.098 percent), which is **11 of 117,090 cleaned
values, 0.0094 percent**, against a stop-and-report gate of 0.1 percent. **Gate
passed.** No dataset lost; the arm stays at 149.

The largest single catch is 87 `Cement` records reporting a per-tonne GWP against
a 1 kg declared unit, a factor-of-1,000 declaration error. Then `PrecastConcrete`
6, `RebarSteel` 4 (including one at 2.59e6 kgCO2e/kg),
`SteelSuspensionAssembly` 3, and `Elevators` 2 at 20,812 and 21,945.

Six datasets change and every other one is bit-identical, which is the proof that
name-keyed Dirichlet weights work as intended:

| dataset | n | coeffvar (unweighted) | skewness | excess kurtosis |
|---|---|---|---|---|
| `Aggregates` | 385 -> 384 | 13.601 -> 6.424 | 19.03 -> 14.92 | 368.3 -> 246.7 |
| `Chairs` | 88 -> 86 | 3.904 -> 2.927 | 6.14 -> 7.66 | 40.3 -> 65.6 |
| `Elevators` | 22 -> 20 | 3.159 -> 1.838 | 3.06 -> 2.73 | 8.1 -> 8.6 |
| `SteelSuspensionAssembly` | 99 -> 98 | 0.178 -> 0.155 | 2.60 -> 2.18 | 10.3 -> 9.0 |
| `Cement` | 1088 -> 1084 | 0.428 -> 0.424 | 1.88 -> 1.92 | 10.6 -> 10.9 |
| `AluminiumExtrusions` | 37 -> 36 | 0.842 -> 0.818 | 2.48 -> 2.52 | 7.1 -> 7.2 |

Arm level: maximum weighted coefficient of variation 14.341 -> 13.404, maximum
excess kurtosis 835.3 -> 525.2, maximum skewness 28.36 -> 21.00. **Medians move
by at most 0.024 on any characteristic.**

**It improves the coverage shortfall without being aimed at it.** Uncovered
dataset-metric pairs 11 of 1,490 -> 10; uncovered on dispersion 5 datasets -> 4,
because `Elevators` is now inside the synthetic range. The remaining four are
`PowerCabling` 13.40, `Aggregates` 7.11, `Grouting` 4.17, `Chairs` 2.89 against a
synthetic maximum of 2.58. **A correction to entry 34 and to the canonical
numbers block: both attribute the arm maximum of 14.341 to `PowerCabling`. It was
`Aggregates`.**

### 4.2 The calibration gate did NOT pass, and nothing was done about it

The tuning objective of `corpus_2026-09-14d` against the arm moves
**0.2075 to 0.2147, which is 0.0072, or 1.10 of the 0.0066 generator
seed-to-seed standard deviation.** Marginally outside. Reported and not acted on:
generation is closed by decisions 47 and 48, and reopening it is an author
decision, not this stage's.

**Three reasons the 1.10 sd overstates the movement**, each measured, so that the
author can decide whether there is anything here at all.

1. **The 0.0066 gate does not cover the term that dominates.** It is GENERATOR
   seed-to-seed noise with the empirical arm held fixed. Six datasets change
   LENGTH under the ceiling, so their Dirichlet weight vector is redrawn from
   scratch; the other 143 are bit-identical. `r3_weight_realization_noise.py`
   measures the missing term directly -- the same arm, the same values, the same
   corpus, twelve independent weight realizations -- at **0.0082**, larger than
   the generator noise itself. Against the two combined in quadrature, 0.0105,
   the movement is **0.69 sd**.
2. **The biggest single contributor is the characteristic the redraw hits.**
   `w_v_uw_wasserstein` moves +0.0446 of the +0.0072 weighted total, and it is
   the metric discrepancy entry 32 records as moving by up to 1.02 per dataset on
   a redraw of the same weights.
3. **The second biggest is a shrinking denominator, not a worse fit.**
   `w1_standardized` divides by the EMPIRICAL standard deviation of the
   characteristic. Removing the `Aggregates` outlier takes the arm's standard
   deviation of `coeffvar` from 1.6467 to 1.2927, down 21.5 percent, while the
   standardized W1 rises only 8.4 percent. In ABSOLUTE terms the coefficient-of-
   variation match IMPROVED, 0.4515 to 0.3843.

**There is nonetheless a real systematic component.** Across the twelve shared
weight seeds the ceiling raised the objective in 11 of 12, mean +0.0064. So the
honest statement is: the arm is marginally harder to match after the ceiling, by
about one generator-seed standard deviation, and a large part of that is an
artifact of the standardizing denominator rather than a worse corpus.

**Recommendation, for the author and not for a later stage to take on itself: do
nothing.** Decision 48 already closes retuning on the grounds that four retunes
all moved the objective by less than one seed standard deviation and two made it
worse. This is a fifth movement of the same size.

### 4.3 Fixtures re-frozen, twice

Once in `e004db4` for the corpus and the ceiling, once in this commit for the
fitting change. `SHA256SUMS.txt` updated in each.

`TABLE_EmpiricalECCMetricsAndW1.xlsx` 138 x 26 -> 149 x 28.
`TABLE_SyntheticECCMetricsAndW1.xlsx` 10,000 x 26 -> 9,999 x 37, the first time
it has been built from a post-regeneration corpus.

**The two synthetic regression tests were structurally broken and are fixed.**
They recomputed from `DATA_all.json`, the retired pre-regeneration baseline, and
compared it to a fixture built from the active corpus: two different datasets,
so they could only ever fail. They now read `CORPUS.json`'s active corpus. They
also carried their own copy of the fitting block and of the scoring grid, which
is exactly the duplication Stage 1 removed from the notebooks; they now drive
`src/fitting.py` directly, so an edit to the fitting method fails the suite
without a notebook being run.

### 4.4 The lognormal

**Which problem the offset was solving: near-zero values.** 14 of 149 datasets
(9.4 percent) still hold a value below 1 percent of their mean, down from Stage
1's 28.3 percent of 138 -- the symmetric log-space cleaning and the plausibility
ceiling account for the difference, and they have not gone away.

| | n | offset beats no-offset 2-parameter | median gain | mean W1, 2p -> offset |
|---|---|---|---|---|
| datasets WITH a near-zero value | 14 | 10 (71.4 pct) | +36.8 pct | 0.813 -> 0.445 |
| datasets WITHOUT one | 135 | 70 (51.9 pct) | +1.4 pct | 0.169 -> 0.156 |

On the 135 clean datasets the offset is a coin flip. That is the answer.

**But it was doing a second thing too, and this is why dropping it outright fails.**
Removing the near-zero values from the FIT takes the no-offset mean W1 on those
14 from 0.813 to only 0.578, while the offset reaches 0.445. Shifting the
threshold to -0.5 also drags the family toward its normal limit, which is a crude
fixed-value version of estimating the threshold. Several of the datasets the
offset helps most have no near-zero value at all: `Tables` +58 percent with a
minimum at 0.0101 of its mean, `AccessFlooring` +62 percent at 0.0131.

**The pathology, demonstrated rather than asserted.** An unguarded joint
optimizer over (threshold, mu, sigma) drives the threshold to within 1e-3 of a
standard deviation of `min(x)` on **21 of the 149 datasets (14.1 percent)**, with
the fitted sigma reaching 11 to 24 against a guarded 2.0 to 2.5; mean W1 0.248
unguarded against 0.182 guarded. `Elevators` reaches a gap of 8.4e-21 and a sigma
of 13.25.

**The guard binds often and is reported, not hidden.** Profile outcome at the
final `PROFILE_DELTA_LO_FRAC = 0.25`, over 149 datasets x 2 weightings: 124
interior (41.6 percent), **148 at the guard (49.7 percent)**, 26 at the normal
limit (8.7 percent). Synthetic, 9,999 x 2: 11,469 interior (57.4 percent), 5,582
at the guard (27.9 percent), 2,947 at the normal limit (14.7 percent).

**For about half the empirical arm the likelihood does not identify a threshold
at all**, and it is set at a fixed fraction of a standard deviation below the
smallest observation. See section 4.9 for why the guard is where it is, and for
why the paper must describe that as a scale-aware version of the heuristic the
offset was rather than as an estimate. **Stage 2h sweeps it where it would have
swept the offset.**

**The two-parameter lognormal is not the simple answer.** Mean W1, empirical arm,
variable weights: two-parameter 0.230, offset 0.183, profile three-parameter
0.182, gamma 0.191. It is the worst of the three lognormals and worse than gamma.
Dropping the offset without replacing it would have made the lognormal worse.

**What the switch moved.** Only the two Lognormal columns; every other column of
the empirical table is identical to 0.000e+00.

| | Stage 1 | Stage 2b |
|---|---|---|
| empirical `Lognormal, Variable`, median W1 | 0.1367 | **0.1220** (-10.8 pct) |
| empirical `Lognormal, Variable`, mean W1 | 0.1785 | 0.1778 |
| empirical `Lognormal, Uniform`, median W1 | 0.1647 | 0.1538 |
| empirical `Lognormal, Uniform`, mean W1 | 0.2056 | 0.2106 |
| synthetic `Lognormal, Variable`, median W1 | 0.0819 | **0.0713** (-13.0 pct) |
| synthetic `Lognormal, Variable`, mean W1 | 0.1104 | **0.0988** (-10.5 pct) |
| synthetic `Lognormal, Uniform`, median W1 | 0.1269 | 0.1211 |

### 4.5 Which method actually wins

**Mean rank over the six methods, and this is finding 1.**

| method | empirical, 149 | synthetic, 9,999 |
|---|---|---|
| `Lognormal, Variable` | **2.09** | 2.19 |
| `KDE, Variable` | 2.74 | **2.12** |
| `Lognormal, Uniform` | 3.34 | 4.17 |
| `KDE, Uniform` | 3.52 | 3.93 |
| `Normal, Variable` | 4.21 | 3.64 |
| `Normal, Uniform` | 5.11 | 4.95 |

**The lognormal leads the empirical arm outright, and on the synthetic corpus the
KDE's lead is now 0.06 of a rank.** Under the Stage 1 method the same ranks were
2.27 against 2.59 empirical and 1.98 against 2.41 synthetic, so refitting the
lognormal widened its empirical lead from 0.32 to 0.65 and cut the KDE's
synthetic lead from 0.42 to **0.06**. On MEAN W1 the synthetic lead is already
gone: `Lognormal, Variable` 0.0988 against `KDE, Variable` 0.1017. The KDE keeps
the synthetic MEDIAN, 0.0635 against 0.0713.

**Fitting by the criterion we score by makes it sharper.** Median reduction in W1
from fitting by W1 rather than by likelihood, empirical arm: normal 31.3 percent,
three-parameter lognormal 12.4, two-parameter 9.2, offset 8.3, gamma 7.7,
improving 85 to 93 percent of datasets in every family.

Mean W1, variable weights, every family at both estimators:

| family | empirical, MLE | empirical, W1-optimal | synthetic, MLE | synthetic, W1-optimal |
|---|---|---|---|---|
| lognormal, 3-parameter profile | **0.178** | **0.138** | **0.100** | **0.081** |
| lognormal, +0.5 offset | 0.183 | 0.151 | 0.111 | 0.096 |
| gamma | 0.191 | 0.149 | 0.115 | 0.099 |
| lognormal, 2-parameter | 0.230 | 0.153 | 0.145 | 0.098 |
| **KDE** | **0.251** | n/a | **0.103** | n/a |
| normal | 0.436 | 0.167 | 0.175 | 0.105 |

**On the empirical arm every parametric family beats the KDE once fitted by the
criterion it is judged by, and three of five beat it even under maximum
likelihood. On the synthetic corpus the three-parameter lognormal now beats the
KDE on mean W1 under maximum likelihood, and under W1-optimal fitting so does
every family except the normal.** The KDE's remaining hold is the synthetic
MEDIAN: 0.063, second to the W1-optimal three-parameter lognormal's 0.057 and
ahead of its maximum-likelihood 0.072.

`FIT_METHOD = 'mle'` remains the study's estimator: maximum likelihood is what a
practitioner would do, and W1-optimal fitting is the control that answers the
objection. Both are implemented and both are reported. Entry 42.

**The caveat that must travel with all of this.** W1 is an IN-SAMPLE criterion
with no complexity penalty, and the families differ in flexibility: normal 2
parameters, two-parameter lognormal 2, gamma 2, three-parameter lognormal 3, KDE
effectively n. Fitting by W1 sharpens that, because the more flexible family
gains more. **Stage 2c owns the held-out or cross-validated version and should
repeat this comparison there before any of it goes in the paper.**

### 4.6 The truncation, and what Stage 1 was really scoring

**The explicit truncation moves nothing.** `r6_method_switch.py` applies one
change at a time, and the "+ grid open at zero" and "+ explicit truncation"
columns are IDENTICAL to every printed digit, on both arms and all six methods.
The reason is that the Stage 1 grid started at exactly zero, so discretizing the
model onto it already truncated and renormalized. Measured directly: the Stage 1
W1 differs from the truncated score by a median of **0.118 percent** and from an
untruncated score by **2.948 percent**.

Opening the grid at zero moves the reported values by +0.17 to +0.37 percent of
the median.

**The normal does not gain from renormalization on the empirical arm, and the
reason is worth stating.** The author expected it to improve, because the model
stops being charged for mass in a region it can never occupy. Measured exactly,
on one lattice spanning both supports: median W1 for `Normal, Variable` 0.2065
untruncated -> 0.1970 truncated, an improvement, and **62 percent of datasets
improve** -- but the MEAN goes 0.4139 -> 0.4334, worse. The mechanism is that the
estimator is the weighted mean and standard deviation, which targets the
UNTRUNCATED moments; renormalizing moves the mass below zero up into the
admissible region and pushes the model's mean above the data's. Where the normal
puts a lot of mass below zero the shift is large, and on the empirical arm it
puts more than 10 percent of its mass there in **41.6 percent of cases** and up
to 47 percent. On the synthetic corpus, where only 18.8 percent exceed 10
percent, both the mean and the median improve.

**So the honest version of the claim the objection deserves** is not "the
parametric families were set up to lose by being charged for mass below zero" --
they were not, the grid was already handling it -- but "the normal's ESTIMATOR
targets untruncated moments while the model is used truncated". Fitting the
normal by W1 closes that, and it is the largest single gain of any family at 31.3
percent.

### 4.7 The empirical arm was scored on unnormalized values

Finding 2, and it is not a Stage 2b change: it is a Stage 1 defect that Stage
2a-2 fixed as a side effect and nobody recorded. The Stage 1 empirical W1 column
was computed on the RAW ECC values while the metrics beside it in the same file
were computed on values divided by their unweighted mean. `WindTurbines` carries
`Normal, Uniform` = 164.26 against a `mean_uw` of exactly 1.0. Reproduced
exactly, and 164.263 / 910.7 = 0.1804, which is the normalized value.

Mean of that column across the arm: **9.38 then, 0.48 now.** Every empirical W1
magnitude the manuscript reports is in the raw unit of its own category, so a
concrete dataset near 400 kgCO2e/m3 and a cement dataset near 0.75 kgCO2e/kg sat
on the same axis three orders of magnitude apart for reasons unrelated to fit
quality.

**The per-dataset RANKS are unaffected**, because W1 is exactly linear in a
rescaling and all six models are fitted to the same values, so every reported
mean RANK is sound. The VALUES cannot be converted -- the new numbers are not the
old ones times any single constant -- and must come from the rerun. Entry 41.

### 4.8 The GWP filter, a count for the manuscript

The extraction keeps only records with a strictly positive ECC. That excludes
**270 records across 57 of the 138 queried categories: 48 exactly zero and 222
negative.** Largest groups `Carpet` 44, `Timber` 18,
`DampproofingAndWaterproofing` 16, `BlanketInsulation` 16, `CMU` 13.

Some are real: biobased products can be legitimately carbon negative over a
boundary that credits biogenic uptake, and the negatives cluster exactly there --
`Timber` 18, `WoodFlooring` 7, `MassTimber` 5, `NonStructuralWood` 4,
`WoodDoors` 4, `CompositeLumber` 3, `HeavyTimber` 2, `WoodFraming` 1. Others are
plainly errors, such as `SheathingPanels` at -12,105 kgCO2e/m3. The filter does
not distinguish them.

**This is a COUNT and not a change.** The filter is unaltered. It belongs in the
paper beside the support statement, because the arm is truncated at zero by
construction and no conclusion about the lower tail of an ECC distribution
generalizes to biobased products. Entry 36.

## 5. Open questions and flags

### Carried forward

Every still-open item from every earlier handoff, restated. An item leaves this
list only by being marked resolved, with the reason.

| Item | Owner | Status |
|---|---|---|
| Bandwidth rule, KL1/KL2 inconsistency | 2h | STILL OPEN |
| `logfit_offset` | 2b | **RESOLVED.** Retired. The profile-likelihood guard `PROFILE_DELTA_LO_FRAC` is what 2h sweeps in its place. Decision 51, section 4.4 |
| Dependent sampling | 2e | STILL OPEN. `rvs_from_uniform` is now in place for it |
| Overlap area alongside W1 | 2c | STILL OPEN |
| Shapiro-Wilk vs Shapiro-Francia | 2f | STILL OPEN |
| `(1-capecc)` divisor | 2g | STILL OPEN |
| Scoring grid includes zero | 2c or 2e | **RESOLVED in 2b.** `score_grid_open`. Entry 18. Decision 50 |
| W1 has no complexity penalty | 2c | STILL OPEN, and **sharper now**: W1-optimal fitting gains most for the most flexible family. Section 4.5 |
| `weighted_quantile` must stay fixed before Silverman in 2h | 2h | STILL OPEN |
| Entry 13, support (0, inf), needs author confirmation | - | **RESOLVED.** Confirmed by the author in the Stage 2b prompt and implemented. Decision 50 |
| Deduplicated empirical variant | 2h | STILL OPEN. Primary stays EPD-level uniform |
| `mode_share_alpha` at 10 | 2h | STILL OPEN |
| `trunc_iqr_mult` sweep | 2h | STILL OPEN |
| Kurtosis undefined in stratum 1 | 2f | STILL OPEN |
| `min_mode_sd_frac = 0.15` has no empirical anchor | 2h | STILL OPEN |
| Six or more modes, 5.6 pct of corpus vs 0.7 empirical | 2h | STILL OPEN |
| `SUPP_DatasetExamplesByStratum.png` x-axis is misleading | 3 | STILL OPEN |
| Figure sizes, git history | 3, 4 | STILL OPEN, untouched. See the new note below on notebook 2 carrying no outputs |
| `audits/stage2a/a6_empirical_source.py` refers to `mode_count_est` | - | STILL OPEN, harmless |
| Notebooks 2 and 3 never run against the active corpus | 2b | **RESOLVED.** Both run clean end to end. Section 3.2 |
| Coverage claim is false at the top of the coefficient of variation | manuscript | STILL OPEN as a TEXT edit, option A, decision 48. The ceiling improved it from 11 uncovered pairs to 10 and from 5 uncovered datasets to 4; entry 34's list and the canonical block both need the new numbers and the `Aggregates`/`PowerCabling` correction |
| A single Dirichlet realization moves per-dataset weighted metrics a long way | 2h | STILL OPEN, and **now quantified at the ARM level too**: the tuning objective's spread over weight realizations alone is 0.0082, larger than the 0.0066 generator seed noise. Section 4.2 |
| Four EC3 parent categories kept as residual bins | - | OPEN as a stated limitation |
| `CementGrout`, `FlowableFill`, `OilPatch` carry a strength field and are not split | - | OPEN by choice |
| EAF against BOF steel is not available | - | CLOSED as infeasible |

### New in Stage 2b

- **THE KDE IS NOT THE BEST METHOD ON THE EMPIRICAL ARM.** Section 4.5, entries
  40 and 42. **This is the one item here that changes what the paper concludes**,
  and it needs an author decision about how to frame the result. It was already
  true before this stage; refitting the lognormal did not create it.
- **The empirical W1 values in the manuscript are in raw category units.**
  Section 4.7, entry 41. Text must take the new numbers from the rerun; they
  cannot be converted.
- **The calibration gate is marginally outside noise and nothing was done.**
  Section 4.2. **Author decision if any; the recommendation is to do nothing.**
- **The ICE database figure in `MASS_ECC_CEILING`'s docstring is unsourced in
  this repository.** Verify before it goes in the paper. Section 3.1.
- **Extremes no external bound can rule on, for author review.**
  `outputs/tables/stage2b/TABLE_2b_UnitExtremes.csv`. Two that stand out: a
  ceramic fibre blanket at 2,300 kgCO2e/m2 in
  `BlanketInsulation [mineral wool]`, 1,564 times its category median, and two
  `ReadyMix [4000-4999 psi]` mixes at 109,292 and 97,859 kgCO2e/m3 against a
  median near 335. **No stage owns these.**
- **The profile-likelihood guard binds in 22.1 percent of empirical fits.**
  Section 4.4. Reported per fit, and **2h owns the sweep**.
- **W1's lack of a complexity penalty is now load-bearing.** Section 4.5. **2c
  must repeat the family comparison out of sample** before any of it is written
  up.
- **The W1 grid is LINEAR and therefore coarse near zero on wide-range
  datasets.** Section 3.5. Unchanged from Stage 1, not introduced here. **2c
  owns the evaluation target.**
- **Notebook 2 carries no stored outputs at all** (0 of 61 code cells), while
  notebooks 1 and 3 carry partial ones. Pre-existing, and the reason it is worth
  fixing is the author's own: the notebooks are the entry point because inputs
  and outputs are visible inline. The obstacle is `figure.dpi = 1200`, which
  makes one executed copy of notebook 2 59 MB. **Same root cause as the figure
  size problem, so: Stage 3.**
- **`customstats.weighted_lognorm_fit` is now unused by the production path** and
  its "MLE" branch numerically re-derives its own "MoM" branch. Entry 38. Left in
  place because the Stage 1 comparison path still calls it.

## 6. Inputs and outputs

**Read:** `CLAUDE.md`, `CONTEXT.md`, `reports/` in full, `src/`, all three
notebooks, `../EPDsFromEC3/store/epd_index.csv.gz` (for the non-positive GWP
count only).

**Written:** `src/families.py`; `audits/stage2b/` (README, r1 to r6);
`tests/test_families.py`; `outputs/tables/stage2b/`;
`reports/HANDOFF_stage-2b.md`.

**Modified:** `src/fitting.py`, `src/empirical.py`, `src/corpus.py`;
`notebooks/01`, `02`, `03`; `tests/test_regression.py`,
`tests/fixtures/TABLE_EmpiricalECCMetrics.xlsx`,
`TABLE_EmpiricalECCMetricsAndW1.xlsx`, `TABLE_SyntheticECCMetricsAndW1.xlsx`,
`SHA256SUMS.txt`; `CLAUDE.md` (decision 13 confirmed, decisions 49 to 53, the
roadmap); `CONTEXT.md` (section 2 rewritten, layout, runtimes, the smoke note,
the test table); `reports/MANUSCRIPT_discrepancies.md` (entries 35 to 42);
every table and figure the three notebooks write.

**Not touched:** the generator, `genconfig.py`, the corpus, the empirical
extract, the manuscript.

**New tables for the author:**

| file | what |
|---|---|
| `TABLE_2b_UnitExtremes.csv` | ten highest and ten lowest records per declared-unit type, with names and ratio to category median. **Needs an eye** |
| `TABLE_2b_PlausibilityDropped.csv` | what the ceiling removed, by dataset |
| `TABLE_2b_PlausibilityEffect.csv` | the six datasets' characteristics before and after |
| `TABLE_2b_NonPositiveGWP.csv` | the 270 records the GWP screen excludes, by category |
| `TABLE_2b_OffsetDiagnosis.csv` | per dataset: near-zero count, W1 with and without the offset, with and without the near-zero values in the fit |
| `TABLE_2b_ThresholdPathology.csv` | per dataset: unguarded against profile threshold, sigma and W1 |
| `TABLE_2b_FamilyComparison.csv` | five families x two weightings x two estimators, both arms |
| `TABLE_2b_TruncationEffect.csv` | normal and KDE, truncated against untruncated, exactly |
| `TABLE_2b_MethodSwitch.csv` | the six methods, one change at a time from Stage 1 to Stage 2b |
| `TABLE_2b_CalibrationAfterCeiling_2026-09-14d.csv` | the gate |
| `TABLE_2b_WeightRealizationNoise_2026-09-14d.csv` | the objective over twelve weight realizations |

## 7. Next stage

**Stage 2c, the evaluation target.** It inherits more from this stage than the
roadmap anticipated, and two of its items are now load-bearing rather than
optional.

1. **Repeat the family comparison OUT OF SAMPLE.** Section 4.5 says every
   parametric family beats the KDE on the empirical arm once fitted by W1. W1 is
   an in-sample criterion with no complexity penalty and the families run from 2
   parameters to effectively n, so that result cannot go in the paper until it
   has a held-out or cross-validated version. `fitting.fit_family(..., 'w1')` and
   `audits/stage2b/r5_family_comparison.py` are the machinery; the missing piece
   is the split.
2. **Score against the KNOWN PARENT**, which is 2c's own headline task and is now
   the cleanest way to settle which method is best: the corpus carries
   `parents.json.gz`, so the synthetic arm has a truth to score against and does
   not need an in-sample criterion at all.
3. **The W1 grid is linear.** Section 3.5. Its resolution near zero is the same
   for every dataset, which is coarse on a dataset spanning orders of magnitude.
   `fitting.score_w1_exact` exists to measure the discretization error.
4. **Overlap area alongside W1**, the item 2c already owned.

**What Stage 2c must NOT do.** Reopen generation; change the fitting families,
which are settled by decisions 51 and 52; or take the calibration-gate decision
of section 4.2, which is the author's.

**What is now available that was not.** Every model exposes `cdf`, `ppf` and
`rvs_from_uniform` on a common support, so a held-out score, an overlap area, a
probability-integral-transform check and Stage 2e's common random numbers are all
one-liners rather than new machinery.

### Read this before touching the fitting again

The four habits in `reports/HANDOFF_stage-2a2.md` section 7 and the two in
`HANDOFF_stage-2a3.md` still hold. This stage adds two.

7. **Check what the code does before explaining why it does it.** Two of this
   stage's four main findings -- that the offset could not have been patching
   the threshold pathology, and that the Stage 1 grid was already truncating and
   renormalizing -- came from reading the code rather than the Methods text, and
   both reversed the natural reading. A third, the unnormalized empirical W1,
   came from noticing that a fixture value was impossible and reproducing it.
8. **A scale error hides from ranks.** The empirical W1 column was wrong by a
   factor of the category's own units for the whole of Stage 1, and every
   rank-based summary of it was correct throughout, because W1 is linear in a
   rescaling. Nothing that looked at ranks could have caught it. When a quantity
   is reported both as a value and as a rank, check the value.

### 4.9 A model can score well on W1 and be useless in the pLCA

**This is the most transferable thing in the stage and it was nearly missed.**

The profile-likelihood lognormal was first implemented with the threshold guard
at 0.01 of a weighted standard deviation below `min(x)`, on the reasoning that a
guard only has to stop the likelihood diverging. Every fit score looked fine.
The defect appeared in the pLCA results: `eci_std` for `Lognormal, Uniform` had a
99th percentile of 33 and a maximum of 532, against 0.92 and 1.56 under the
Stage 1 method, and `eci_mean` reached 9.6 on data whose mean is exactly 1.0.

**The mechanism.** Where the profile likelihood has no interior maximum the
threshold is driven onto the guard, and the closer the guard sits to `min(x)`
the larger sigma must be to accommodate the rest of the data. The fitted
lognormal then matches the BODY of the data and carries an enormous right tail.
**W1 barely charges for that**, because a thin far tail is a small area between
two CDFs. The pLCA does not care: it SAMPLES from the model, and a model with a
standard deviation of 3,000 dominates any Monte Carlo it enters.

Measured across both arms, the largest standard deviation of any fitted model,
on data whose own standard deviation is near 0.6:

| guard, in sd below min(x) | empirical max model sd | empirical mean W1 | synthetic max model sd | synthetic mean W1 | pct at the guard, empirical |
|---|---|---|---|---|---|
| 0.01 | **3281** | 0.1815 | **5345** | 0.1037 | 22.2 |
| 0.05 | 77 | 0.1699 | 99 | 0.0995 | 32.9 |
| 0.10 | 18 | 0.1691 | 19 | 0.0979 | 38.9 |
| **0.25** | **3.4** | **0.1778** | **2.9** | **0.0975** | **48.3** |
| 0.50 | 1.9 | 0.1941 | 1.5 | 0.1010 | 59.1 |
| 1.00 | 1.7 | 0.2203 | 1.3 | 0.1087 | 69.8 |

**`PROFILE_DELTA_LO_FRAC = 0.25`**, chosen as the smallest guard at which no
fitted model has a standard deviation above five times the data's, on either
arm -- zero such datasets, against 20.1 percent of the empirical arm at 0.01.
**It is chosen on the bounded-variance criterion and NOT on W1**, deliberately,
so that it is not a number tuned to the score it is then judged by; W1 is flat
from 0.05 to 0.25 and better there than at 0.01 on both arms, so the choice
costs nothing. `audits/stage2b/r7_profile_guard.py`,
`outputs/tables/stage2b/TABLE_2b_ProfileGuardSweep.csv`.

**What the paper must say about it.** At 0.25 the guard determines the threshold
for **48 percent of empirical fits** and 30 percent of synthetic ones. For about
half the empirical datasets the likelihood does not identify a threshold at all,
and it is set at a fixed fraction of a standard deviation below the smallest
observation. **That is a scale-aware version of exactly the heuristic the +0.5
offset was**, and it should be described as one rather than presented as an
estimate. It is a better heuristic -- it adapts to the dataset's scale and it is
the boundary of a real likelihood problem rather than a round number -- but it is
a heuristic.

**Two alternatives were measured and rejected.** On the 116 guard-bound datasets,
falling back to the two-parameter lognormal is WORSE, not better: median model
standard deviation 10.4 on the empirical arm and a maximum of 1.6e5, because a
two-parameter lognormal fitted to data with a value at 1e-5 of its mean has a
huge sigma of its own. Falling back to the +0.5 offset fit gives bounded models
(max sd 1.79) and so does gamma (max 2.20), and gamma has the best W1 of the four
on those datasets. Both would make the estimator a hybrid, which is harder to
defend than one method with one stated guard. **If the author prefers a hybrid,
gamma is the one the measurements favour.**

**`tests/test_families.py::test_profile_fit_is_a_usable_generative_distribution`
now asserts it.** That test is the one that would have caught this from the fit
alone. The general lesson, and it applies to every method this study scores:
**a goodness-of-fit statistic between CDFs is nearly blind to tail mass, and the
pLCA is not.** Stage 2c owns the evaluation target and should decide whether W1
alone is enough; Stage 2g owns the downstream metrics and inherits the same
question from the other end.

### 4.10 What the fitting change did to the pLCA

Two full pLCA runs, 2,499 groups x 6 methods x 4 datasets = 59,976 rows each,
the same seed and the same corpus, differing only in the fitting method.

**The Normal and the KDE moved by exactly the Monte Carlo floor and no more.**
Mean absolute movement in `eci_rank_1`, the headline metric: `KDE, Uniform`
0.0048, `KDE, Variable` 0.0046, `Normal, Uniform` 0.0047, `Normal, Variable`
0.0045, against a Monte Carlo standard error of 0.0047 at `neccs = 10000`
(discrepancy entry 17). Those two methods are the same distributions as before;
what changed is that the draws come from an inverse CDF rather than from
rejection, so they are different draws from the same distribution. **That is the
cleanest available confirmation that the truncation is a restatement and not a
new model.**

**The Lognormal moved by about 3.5 times the floor**, 0.0166 and 0.0163, which is
the refit. `eci_rank_1`'s standard deviation across datasets moves 0.1128 to
0.1202 for `Lognormal, Variable` and is flat everywhere else.

**And the tail check, which is section 4.9's defect seen from the pLCA end:**

| `eci_std`, `Lognormal, Uniform` | Stage 1 | guard 0.01 | guard 0.25, shipped |
|---|---|---|---|
| median | 0.555 | 0.608 | 0.553 |
| 99th percentile | 0.924 | **33.1** | 1.198 |
| maximum | 1.558 | **532.3** | 1.911 |
| `eci_mean` maximum | 1.236 | **9.567** | 1.220 |

## 8. The stage's own assessment

The reference point is `reports/HANDOFF_stage-0.md` section 8, which is not
edited.

**Analysis quality: better, and for a reason worth naming.** The stage set out to
fix the lognormal and found four things that were wrong in ways nobody had
looked for: the Methods text describes a fit the code has never performed; the
empirical W1 column was in raw category units for the whole of Stage 1; the pLCA
had been silently dropping three datasets; and a model can pass the study's own
goodness-of-fit criterion while being unusable as the sampler the study then
makes it. Three of the four were found by reading code or by noticing an
impossible number, not by a test.

**Code quality: better in the part that was touched, and one duplication was
found and removed that Stage 1 had missed.** `tests/test_regression.py` carried
its own copy of the fitting block and of the scoring grid, so it could not have
detected a change in the code it was meant to be guarding. It now drives
`src/fitting.py`. The test count is 128 at the end of Stage 1 and 238 now.

**What is NOT better.** `src/customstats.py` is still a flat module of mixed-era
helpers with a commented-out import line at the top as its table of contents, and
`weighted_lognorm_fit` is now dead weight in the production path. Notebook 2
still carries no stored outputs. Neither is this stage's scope and both are
recorded above.

**The honest risk this stage leaves.** The lognormal's threshold is set by a
guard rather than by the data for half the empirical arm. That is defensible and
it is stated, but it is a heuristic in a paper whose contribution is a comparison
of methods, and a reviewer who reads section 4.9 will ask why the three-parameter
lognormal is used at all when gamma has no threshold, no pathology, no guard, and
beats it on the guard-bound datasets. **Stage 2c should answer that question with
the out-of-sample comparison rather than leaving it to the reviewer.**
