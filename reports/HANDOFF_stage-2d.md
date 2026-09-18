# HANDOFF stage-2d - The weighting measures and the flip threshold

US spelling throughout, as in every file this project writes.

**HOW TO READ THIS FILE. Its reader does NOT have this repository** -- no
CLAUDE.md, no other report, no table, no figure, no source. Every claim below is
therefore stated in full where it is made, and a trailing `decision N` or
`entry N` is a citation into the project's decision log or its manuscript
discrepancy log, never the substance of the sentence. Where a file has to be
opened, the instruction is addressed to the NEXT CLAUDE CODE SESSION and says so.
If a sentence here cannot be understood on its own, that is a defect in this
file.

# IF YOU READ ONE PAGE, READ THIS ONE

Everything below is the working record. **These ten sentences are what Stage 2d
contributes to the manuscript.** Each is a claim the paper can make, with the
number that supports it. Nothing else in this file needs to reach the paper.

**On the method comparison**

1. **Switching UQ method changes a material's estimated contribution by about
   18 percent.** For the most-affected material in a typical probabilistic LCA,
   with every material contributing a mean of 1.00: its estimated contribution
   moves by **0.18**, its 95th percentile by **0.41**, its share of the building
   total by **3.0 percentage points**, and its contribution to total variance by
   **8.9 points**. At the 90th percentile those become 0.52, 1.29, 8.5 points and
   27.6 points. The choice of method is not a rounding detail.
2. The output that moves most is the **spread**, not the average: a material's
   standard deviation moves by 0.17 and its 95th percentile by 0.41, against 0.18
   for its mean. That is what one would hope, since representing spread is what a
   UQ method is for, and it means the methods differ most where they are supposed
   to.
3. **The steadiest output is the one that sets data-collection priorities.** A
   material's contribution to total variance moves by 8.9 points against 12.6 for
   its chance of being the largest contributor, and it is the least affected
   output measured. So "which material is worth measuring better" survives the
   choice of method better than any magnitude does -- and the study currently
   computes it and reports it nowhere.
4. The probability that the identity of the largest contributor changes crosses
   **1 percent at a relative W1 of 0.0018, 5 percent at 0.011 and 10 percent at
   0.025**. **A relative W1 of 0.011 is a shift of about one percent of the
   dataset mean** -- invisible on a plotted curve, and enough to change the
   answer one time in twenty.

**On weighting, which is the stage's main contribution**

5. Whether weighting matters is predicted almost exactly by two numbers a
   practitioner already has, the EPD count `n` and the coefficient of variation
   `CV` (the standard deviation divided by the mean): **the separation is about
   0.73 * CV * n^-0.43**, R2 = 0.99 on both arms.
6. Uniform weighting is therefore safe only when the coefficient of variation is
   below about **0.015 * n^0.43** -- 0.046 at 10 EPDs, 0.120 at 100, 0.315 at
   1,000 -- and the median real category, at CV 0.63 and 47 EPDs, does not clear
   it.
7. **For 91 percent of real ECC categories the market-share assumption is not
   safe**, in the sense that a plausible allocation moves the fitted density
   further than the distance at which UQ methods begin to change the answer.
   **This is the paper's call to collect market-share data**, which is the one
   input that would remove the problem rather than bound it.
8. Reweighting acts mostly on the MEAN -- median location share 0.725 -- but that
   is a median and not a rule: the interquartile range is **0.457 to 0.933**, and
   **28 of 147 categories are shape-dominated**, where reweighting changes the
   distribution's shape and barely moves its mean. Which way a category behaves is
   **not predictable** from its size or its dispersion (Spearman -0.11 and -0.10).
   The useful consequence is that for most categories the uncertainty from unknown
   weights is uncertainty about a mean, which is exactly what a production-weighted
   industry-average EPD would supply.
9. Every weighting number here is a **lower bound**, because a flat Dirichlet
   understates the separation by **1.5 to 3.1 times** when share clusters on
   similar products, which is how real market share behaves.
10. This is the one place in the whole study where **dispersion matters as much as
    dataset size**; everywhere else size is the only mechanism.

**What the paper must state as conditional, in the same paragraph as the number
and not in a footnote.** Every FLIP probability assumes four materials of equal
material use intensity, which makes a ranking as fragile as it can be made and
therefore makes those numbers upper bounds. The continuous outputs in sentences 1
to 3 do not have that dependence, which is the reason to lead with them. Stage 2e
changes the number of materials per pLCA and will move every flip threshold.

**One housekeeping item, not a finding.** The plausibility ceiling's inventory
citation could not be verified in this repository and is withdrawn. Nothing about
the ceiling changes: it rests on the arithmetic that 100 kgCO2e per kg of product
would require burning 27.3 kg of pure carbon per kilogram shipped, which needs no
database and which a reviewer checks in one line.

---


---

## 0. STATUS

**What the stage was asked for.** Decompose the distance between the
uniform-weighted and the variable-weighted version of a dataset into a location
part and a shape part; define and verify a named relative measure; build the
per-dataset statement "assuming uniform weights has an X percent chance of
changing which material ranks first", using A_IQR from the author's own
published paper as the instrument; and calibrate that X against the downstream
decision, giving the relative W1 at which the flip probability crosses 1, 5 and
10 percent. Plus two housekeeping items.

**All of it was delivered. Several things did not come out as expected, and each
is stated plainly below rather than smoothed into the story.** Two further
results came out of the author's review of the first draft and are the most
useful things here for a practitioner: a closed form for when weighting matters,
and a measured limit on the weight model this whole study rests on.

### The results, in the order they matter

**1. SWITCHING UQ METHOD CHANGES A MATERIAL'S ESTIMATED CONTRIBUTION BY ABOUT
18 PERCENT.** Measured over 250 probabilistic LCAs and all fifteen pairs of the
six methods, giving both methods the same draws so the comparison is
like-for-like. Each figure below is the change for the most-affected of the four
materials in a group, which is the one a practitioner is deciding about. Every
material contributes a mean of 1.00 in this study, so these read directly:

    estimated contribution                 0.181     (90th pct 0.519)
    95th percentile of the contribution    0.413     (90th pct 1.290)
    standard deviation of contribution     0.173     (90th pct 0.467)
    coefficient of variation               0.175     (90th pct 0.389)
    chance of being largest contributor    0.126     (90th pct 0.278)
    contribution to total variance         0.089     (90th pct 0.276)
    share of the building total            0.030     (90th pct 0.085)

**The spread outputs move most** -- the 95th percentile by 0.41 against the
mean's 0.18 -- which is the right way round, since representing spread is what a
UQ method is for. **The contribution to total variance moves least**, so "where
should I collect better data" is the steadiest answer a probabilistic LCA gives,
and the study computes it and reports it nowhere. Decision 103.

**2. THE CURVE, which is the deliverable.** The probability that the
top-contributing material changes crosses **1 percent at a relative W1 of
0.0018** (95 percent interval 0.0013 to 0.0023), **5 percent at 0.011** (0.0091
to 0.0126) and **10 percent at 0.025** (0.0217 to 0.0277). An isotonic fit, which
assumes only that the probability does not fall as two models separate, gives
0.0026, 0.0129 and 0.0271. **Quote two significant figures and no more**: an
independent run of the same calculation on a different random stream gave
0.0015, 0.0099 and 0.0233, every one inside the other run's interval and every
one differing in the third figure. Decision 95.

**What that says about the six methods under test:** they sit at a median
separation of 0.30, an order of magnitude past the 10 percent crossing. Result 1
is the better way to say what that costs, in units that do not depend on how many
materials a pLCA holds.

**3. REWEIGHTING USUALLY MOVES THE MEAN, BUT NOT ALWAYS, AND NOTHING PREDICTS
WHICH.** The uniform-to-variable distance splits into a shift of the mean and a
change of shape. The mean term carries a median of **0.725** of it on the 147
real categories -- but that is a median, not a rule: the interquartile range is
**0.457 to 0.933**, and **28 of the 147 are shape-dominated**, below 0.4, where
reweighting rearranges the distribution and barely moves its mean.
`CementGrout` sits at 0.024, `PowerCabling` at 0.077, `RebarSteel` at 0.082.

**Which way a category behaves is predictable from nothing measured here**:
Spearman -0.11 against log dataset size and -0.10 against the coefficient of
variation.

**This is not a recipe and an earlier draft wrongly offered it as one.** Nobody
can compute a market-weighted mean, because EPD market shares are not published,
which is the premise of the companion paper. What the finding says is what KIND
of uncertainty unknown weights introduce -- for most categories, uncertainty
about a mean rather than about a shape. That is why an industry-average EPD, the
one published quantity that is production-weighted, would resolve most of it.
The practitioner rule is result 4, not this. Decisions 92 and 104.

**4. WHAT DECIDES WHETHER WEIGHTING MATTERS IS BOTH SIZE AND DISPERSION, and
together they give a closed form.** This is the one place in the project where
dispersion is a first-order quantity. Regressing the log of the separation on
log dataset size and log coefficient of variation: size alone explains 48 percent
of the variance, dispersion alone 50 percent, and **both together 99.1 percent**,
with each adding about half on top of the other. They are nearly orthogonal. The
fit is

    log(separation) = -0.32 - 0.434 * log(n) + 1.036 * log(CV)

and the same exponents come out of the synthetic arm, -0.427 and 0.977, so the
practitioner form is **separation is about 0.73 * CV * n^-0.43**.

**Set that against the calibrated 5 percent threshold and the rule needs no
distribution at all**: uniform weighting is safe only when the coefficient of
variation is below about **0.015 * n^0.43** -- 0.046 at ten EPDs, 0.120 at a
hundred, 0.315 at a thousand. The median real category has a coefficient of
variation of 0.63 at 47 EPDs, so almost none of them clears it, which is the same
conclusion the per-dataset probabilities reach by a different route.

**5. A FLAT DIRICHLET UNDERSTATES THE WEIGHTING RISK, AND THE NUMBERS HERE ARE
THEREFORE A LOWER BOUND.** The author asked whether uniformly exploring the
simplex is the right model, given that real market share probably arrives in
clusters, and suggested that a cluster is nearly a dataset with fewer points, so
the effective sample size would already capture it. **Half of that is right, and
the half that is not is the important half.** Comparing three weight schemes at
MATCHED effective sample size: concentrating share on randomly chosen products
gives separations 0.90 to 0.99 times a flat draw, which is no difference --
so the effective sample size does capture concentration. Concentrating the same
share on products with ADJACENT coefficients, which is what clustering means in
practice, gives **1.5 to 3.1 times** the separation at the same effective sample
size.

What the effective sample size misses is coherence. A contiguous block shifts the
whole distribution one way, which lands in the location term that already carries
72.5 percent of the uniform-to-variable distance; random concentration moves
mass in directions that partly cancel. **The direction of the error is
conservative for this paper**, whose finding is that uniform weighting is rarely
safe.

### The two housekeeping items

**The ICE figure is unsourced and is withdrawn from the paper.** The physical
plausibility ceiling of 100 kgCO2e per kg, which removes impossible records
before cleaning, was justified partly by a highest building-product coefficient
of about 13 kgCO2e/kg for primary aluminium attributed to the Inventory of Carbon
and Energy database version 3.0. That figure came from an earlier session's own
knowledge; no copy of that database and no other file in the repository contains
it, so the number, the edition and the page are all unverifiable. **Do not cite
it.** The ceiling is unchanged and needs no database: burning pure carbon yields
44.009 / 12.011 = **3.664 kg CO2 per kg of carbon**, so 100 kgCO2e per kilogram
of delivered product would require burning **27.3 kg of pure carbon for every
kilogram shipped**, and even a ceiling of 25 would require 6.8 kg. That is
arithmetic a reviewer checks in one line. Discrepancy entry 81.

**The handoff specification now says who reads a handoff.** It records that the
reader has no repository, that decision and entry numbers are trailing citations
rather than substance, that numbers must appear as text, and that an instruction
to open a file is addressed to the next Claude Code session.

---

## 0b. Four things a reader will ask, answered here so they are not re-derived

**What "relative W1" means.** W1 between two distributions over one dataset,
divided by that dataset's own unweighted mean. Every W1 this study has ever
reported is already this, because every dataset is divided by its unweighted
mean before anything else happens. Naming it changes no number.

**What "CV" means.** The coefficient of variation: the standard deviation
divided by the mean.

**Why the location/shape split is not a recipe.** Nobody can compute a
market-weighted mean, because EPD market shares are not published -- that is the
premise of the companion paper. The split says what KIND of uncertainty unknown
weights introduce, which for most categories is uncertainty about a mean rather
than about a shape. It is a reason to want industry-average declarations, not an
instruction to compute something.

**Why the plausibility ceiling's citation was withdrawn.** The mass ceiling of
100 kgCO2e/kg was justified partly by a figure attributed to the Inventory of
Carbon and Energy database, which could not be verified from anything in the
repository. Nothing about the ceiling changes: it rests on the arithmetic that
100 kgCO2e per kilogram of delivered product would require burning 27.3 kg of
pure carbon per kilogram shipped, which needs no database.

## 1. Stage and branch

| | |
|---|---|
| **Stage** | 2d, the weighting measures and the flip threshold |
| **Branch** | `stage-2d-threshold` |
| **Branched from** | `638ffd9`, on branch `stage-2c-target`. Working tree clean at the branch point |

---

## 2. What was asked

Two housekeeping items: verify or withdraw the inventory figure behind the
plausibility ceiling, and write into the project brief that a handoff's reader
has no checkout.

Then: decompose the uniform-to-variable Wasserstein distance into a location
component and a shape residual and report the ratio; define a named relative
measure, verify its invariance by rerunning un-normalized, and compare it with a
robust-scale version; and calibrate the flip probability against relative W1 by
logistic or isotonic regression, reporting the crossings at 1, 5 and 10 percent
with confidence intervals. The per-dataset weighting risk was to use A_IQR as its
instrument, with two construction details to be read off the published paper
rather than reconstructed.

---

## 3. What was done

**New code, both tested.** A module holding the location/shape split, the named
relative measure with its three candidate denominators, and A_IQR with the
ensemble it comes from. A second module holding the common-random-numbers pLCA,
the model-to-model distances, a logistic and an isotonic fit, and a bootstrap
that resamples pLCA groups. Twenty-eight new tests, all passing; the whole suite
is 369 tests and all pass, including the eight regression fixtures, which is the
check that nothing existing moved.

**New notebook sections.** Notebook 1 gained the decomposition on both arms, the
relative measure with its un-normalized verification, A_IQR and the per-dataset
weighting risk at 1,000 Dirichlet draws, a post-stratified aggregate, and a
three-panel figure. Notebook 3 gained the six method pairs under
common random numbers, the calibration set, the crossings, a provenance check,
post-stratified flip rates and the calibration figure. **Notebook 3's new
material writes no pLCA result and replaces nothing above it.**

**Five audit scripts** measure what the settings are: the ensemble convergence,
the full calibration with its diagnostics, the weighting measures end to end, how
how a tie-break between two near-identical materials behaves, and what
switching UQ method does to every pLCA output. The last of these was stopped part-way through its
final run, when it was competing for processor time with the notebook that
produces the paper's own tables; it writes only to the audit directory, nothing
reported depends on it, and re-running it is a single command. The next Claude
Code session will find it at `audits/weighting_measure.py`.

### Two details read off the published paper

Both were open when the stage began and both were settled from the local copy of
Torres et al. (2026): the area is not normalized, and the ensemble is 1,000
draws. **They no longer matter to this paper**, because A_IQR is dropped from it
(section 4.5), but they are recorded so nobody re-reads the paper to find them.

### Two defects the tests caught, both of which would have been silent

**The isotonic regression dropped a point on every merge.** The pool-adjacent-
violators step was written so that a list element was removed before the index of
its destination was computed, which put the merge on the wrong block. The fitted
curve came back shorter than its input and did not preserve the mean. It now
matches a reference implementation to 1.1e-16 over 500 points.

**The density ensemble was built from a bare kernel sum rather than the study's
own fitted density**, which is a kernel estimate explicitly truncated at zero and
renormalized, because an embodied carbon coefficient of zero or less is not
admissible. On small datasets with a concentrated weight draw the bandwidth is
wide enough that up to 10 percent of the mass fell below zero and off the grid,
so the ensemble would have been partly a measure of how much mass each draw
spilled. It feeds the weighting separation as well, so the fix matters even
though A_IQR itself is now dropped.

---

## 4. What the measurements say

### 4.1 What switching UQ method actually does to a pLCA result

Measured over 250 pLCA groups and all fifteen method pairs, giving both methods
the same uniform draws so the comparison is like-for-like. Each figure is the
change for the MOST-AFFECTED of the four materials in a group, which is the
material a practitioner would be making a decision about.

    output                              median change   90th pct
    estimated contribution                      0.181      0.519
    95th percentile of the contribution         0.413      1.290
    standard deviation of contribution          0.173      0.467
    coefficient of variation                    0.175      0.389
    chance of being largest contributor         0.126      0.278
    contribution to total variance              0.089      0.276
    share of the building total                 0.030      0.085

**Every material contributes a mean of 1.00 in this study, so 0.181 on the first
row is an 18 percent change in a material's estimated contribution**, purely from
which of six defensible UQ methods was used.

Two things worth carrying into the paper. **The spread outputs move most**, which
is the right way round: a UQ method exists to represent spread, so that is where
two of them should differ. And **the variance contribution moves least**, so the
question "where should I spend effort collecting better data" is the most robust
output a probabilistic LCA produces -- more robust than any magnitude it reports.

### 4.2 The decomposition: mostly location

W1 between two distributions is at least the absolute difference of their means.
Here both are the same values under two weightings, so that bound is the
difference between the weighted and the unweighted mean. Call it the LOCATION
component; the rest is SHAPE.

| arm | median location share | mean | pooled | above half |
|---|---|---|---|---|
| empirical, 147 datasets | **0.725** | 0.673 | 0.728 | 68.7 pct |
| synthetic, 10,000 datasets | **0.804** | 0.695 | 0.799 | 72.4 pct |

**By dataset size, on the real data**, the median location share is 0.96 at 3 to
9 EPDs, 0.65 at 10 to 99, 0.74 at 100 to 999 and 0.54 at 1,000 and above. The
inequality holds throughout: the worst residual across all 10,147 datasets is
-7.0e-14, which is floating point in the quadrature and not a result.

**What it means for the paper, and it is NOT a recipe.** Nobody can compute a
market-weighted mean, because EPD market shares are not published -- that is the
premise of the companion paper. What this says is what KIND of uncertainty
unknown weights introduce: for most categories, uncertainty about a mean rather
than about a shape, which is why a production-weighted industry-average EPD would
resolve most of it. For the 28 shape-dominated categories it would not. The rule a
practitioner can actually apply is the size-and-dispersion law in section 4.4a.

### 4.3 The relative measure was already there, unnamed

Every dataset in this study is divided by its own unweighted mean before anything
else happens, so **every W1 the study has ever reported is already a W1 divided
by a mean**. Naming it changes no number.

**Verified rather than asserted.** The same quantity was recomputed on the raw
empirical values, in their own kilograms of CO2 equivalent per declared unit,
with dataset means spanning **0.0685 to 910.7 -- a factor of 13,000**. The
relative measure agrees with the normalized run to **6e-15**. The absolute
distance scales by exactly the dataset's own mean, which is the control that the
check is checking something.

**The two robust denominators were computed and are not adopted.** Dividing by
the interquartile range or by the standard deviation separates the flips
essentially as well: the area under the receiver operating curve, on the
top-contributor outcome, is 0.8018 for the mean, 0.8004 for the interquartile
range and 0.8056 for the standard deviation. Half a percent is not a reason to
change the denominator the study already uses, and the mean is the only one of
the three that a practitioner computes without ambiguity. The standard deviation
is better on the full-ordering outcome, 0.794 against 0.761, which is noted and
not acted on because the full ordering is not a criterion this study reports.

**All three are computed with UNIFORM weights, and that is the decision that
matters here.** A denominator taken under the variable weights would move when
the weights move, which is the quantity being measured; and a practitioner
holding a set of environmental product declarations cannot compute a
market-weighted mean without already knowing the market shares, which is exactly
what they lack.

### 4.4 THE CURVE

Fitted on 2,500 probabilistic LCAs by nine reweighting levels, 20,000
observations, under common random numbers. Logistic regression on the logarithm
of the distance, because the separations span four orders of magnitude and all
three levels sit in the lower part of that range. The interval is a percentile
bootstrap that **resamples pLCA groups rather than rows**, because the
comparisons inside a group share four datasets and one set of random variates and
are not independent observations.

| flip probability | relative W1 | 95 percent interval | isotonic fit |
|---|---|---|---|
| 1 percent | **0.0018** | 0.0013 to 0.0023 | 0.0026 |
| 5 percent | **0.011** | 0.0091 to 0.0126 | 0.0129 |
| 10 percent | **0.025** | 0.0217 to 0.0277 | 0.0271 |

**Two significant figures, deliberately.** The interval is about 30 percent of
the estimate wide, and an independent run on a different random stream gave
0.0015, 0.0099 and 0.0233 -- all inside the intervals above, all differing in the
third figure. Five figures would be false precision and would make the number
drift on every rerun.

The levels being read are inside the data rather than extrapolated to: the
observed flip rate in the lowest bins runs 0.005 at a separation of 0.001, 0.015
at 0.0024, 0.021 at 0.0039, 0.034 at 0.0057 and 0.051 at 0.0093.

**The six UQ methods could not supply this curve, and that is itself a result.**
Over 37,500 comparisons the smallest relative W1 between any two of the six is
0.00022, and the flip rate in the lowest 2 percent of separations is already 14.1
percent. Every level being asked about lies below the observed data;
an isotonic fit on those pairs alone returns the same crossing for all three
levels, because its first block is already above the top of them. The calibration
set therefore adds pairs at controlled separations running continuously to zero:
the same kernel estimate under uniform weights, and under weights moved a
fraction of the way toward a Dirichlet draw.

**The device is checked rather than assumed.** If the curve describes the
DISTANCE rather than how the distance arose, then the six real method pairs --
which are different distribution families -- should fall on a curve fitted from
reweighting pairs. They do over most of the range:

| separation band | calibration set | the six method pairs | pairs in band |
|---|---|---|---|
| 0.012 to 0.021 | 0.064 | 0.028 | 36 |
| 0.021 to 0.036 | 0.109 | 0.063 | 128 |
| 0.036 to 0.064 | 0.176 | 0.180 | 894 |
| 0.064 to 0.124 | 0.281 | 0.288 | 4,595 |
| 0.124 to 0.716 | **0.436** | **0.606** | 29,943 |

The two agree closely in the two bands that hold most of the method pairs. Below
them the method pairs are too few to say anything -- 36 and 128 comparisons, and
fewer than 20 below that.

**They diverge in the top band.** A cross-family difference of a given size is
more consequential than a reweighting difference of the same size, presumably
because the families differ in the tails that decide a ranking. **So the curve
understates the flip probability for large cross-family differences and should be
read as a lower bound there.**

**THE CAVEAT THAT MUST TRAVEL WITH EVERY ONE OF THESE NUMBERS.** Every material
in this study is normalized to a mean of 1.0 and carries a material use intensity
of 1.0, so the four contributions in a probabilistic LCA are nearly exchangeable
and their ranking is as fragile as it can be made. A real building, where
materials differ by orders of magnitude in contribution, is much harder to flip.
**These crossings are an upper bound on how often a modeling choice changes an
answer.** That is the conservative direction for a practitioner rule, but it must
not be quoted as a statement about buildings.

**The full rank ordering is not a usable criterion in this construction**, and is
reported only in order to say so. Its crossings are at 0.00002, 0.00023 and
0.00065. Ranking four near-identical materials from first to last is not a
decision anyone makes.

**Post-stratified**, because the synthetic corpus allocates datasets equally
across four size bands while the real categories do not: the top-contributor flip
rate over the calibration set is **0.1434 at equal allocation and 0.1147
reweighted** to the empirical size mix. Neither is the true one; both are
reported, as every headline in this study is. The unit of a flip is a group of
four materials and therefore has no single size band, so the convention is the
smallest material in the group, on the grounds that the small dataset is where
fitted models differ most. It is stated rather than hidden.

### 4.5 A_IQR is dropped from the paper

**Author decision, 2026-09-17: "that was a hypothesis I had that didn't pan out
... let's stop mentioning it and don't bring it up in the manuscript. It served a
different purpose for a different study."**

The measure stays in `src/weighting.py` and the per-dataset values stay in
`TABLE_WeightingRisk.csv`, so nothing has to be recomputed if it is ever wanted
again. It is simply not part of this paper's argument.

**What was learned before it was dropped, in one paragraph, because it is a real
result about the measure and not a failure of it.** A_IQR in this study is a
function of dataset size: Spearman -0.946 against log n, +0.042 against the
coefficient of variation, and its exponent of -0.35 is what density-estimation
theory predicts, since the standard deviation of a kernel density estimate goes
as n^-0.4 under a Silverman bandwidth. **That does not make it a poor measure.**
It responds to the WEIGHT INFORMATION as well as to n, and the companion paper's
own scenarios show it separating 0.22 from 0.12 on the same nine data points when
the only change is whether a subgroup's shares are constrained or known. This
study holds the weight information fixed for every category -- a flat Dirichlet,
no constraints -- so the only dimension left varying is n. The measure is fine;
this study's design removes the axis it was built to see.

---

## 5. What is still open

### Owned by a later stage

| Item | Owner |
|---|---|
| **Common random numbers in the STUDY's pLCA.** A tested implementation is in `src/flip.py`; giving two methods the same uniform draws makes their comparison exactly paired. Worth doing, and not urgent: every continuous output already separates the methods far more than sampling variation does | 2e |
| Dependent sampling; materials per pLCA swept over 2 to 12; resampled groupings; bootstrap intervals on every headline percentage | 2e |
| **Run the pLCA against the TRUE parent distributions**, which Stage 2c called the most valuable single experiment available. It converts every score from "how close is the fitted CDF" to "how wrong is the answer" | 2e, then 2g |
| Shapiro-Wilk versus Shapiro-Francia; the full reduction of the characteristic set to three to five survivors | 2f |
| The `(1-capecc)` divisor; magnitude-based companion metrics; sensitivity of the headline rank-1 frequency | 2g |
| The profile-likelihood guard sweep, reporting the fitted-model spread ratio at every value; the Dirichlet concentration sweep; multiple weight realizations; the deduplicated variant | 2h |
| **A smoke run reached a commit in this stage.** It was caught and restored within the session and cost nothing, but the rule against it is a sentence in a document rather than a check. The fix is for the notebook to refuse to write its main table when the smoke environment variable is set, or for a test to assert the group count in the run metadata | 3 |
| **Every figure in the repository must be brought to `FIGURE_STYLE.md`.** The author's instruction is that the guide is binding on all figures, not only on the ones Stage 2d produced. Two are compliant; the rest of notebooks 1, 2 and 3 predate the guide and none has been checked against it. `src/figstyle.py` provides the palette, the rcParams, direct labelling and an automatic text-overlap check | 3 |
| Figure rebuilds and the figure manifest; one results table is 96 MB | 3 |

### Text the manuscript owes, from this stage

| | |
|---|---|
| The inventory citation | Withdrawn. State the plausibility ceiling with the stoichiometric justification and no database citation. If corroboration is wanted, someone must open the Inventory of Carbon and Energy version 3.0 and record an edition and a page |
| The relative measure | Say once, explicitly, that every reported distance is relative to the dataset's own unweighted mean. It moves no number |
| The decomposition | Report the median location share as **0.725**, with its interquartile range of 0.457 to 0.933 and the 28 of 147 shape-dominated categories. Do NOT report it as a constant, and do NOT frame it as a recipe: nobody can compute a market-weighted mean |
| A_IQR | **Nothing. It is dropped from the paper by author decision.** Section 4.5 says why; do not reintroduce it |
| The curve | New result and new figure. Report the crossings with their intervals, and report the caveat about exchangeable materials in the same paragraph, not in a footnote |

### Known and accepted

The calibration curve understates the flip probability for large cross-family
separations, as section 4.4 shows and states. The flip levels are an upper bound
because the study's four materials are near-exchangeable. The synthetic arm of the per-dataset weighting table is a stratified sample of
400 rather than the whole corpus, because 1,000 Dirichlet draws per dataset costs
about a second; the sample size is reported beside every number taken from it.

---

## 6. Numbers that moved

**No previously published number moved.** The full test suite, including the
eight regression fixtures that pin the empirical metrics, the synthetic metrics
and all six goodness-of-fit scores, passes unchanged. Everything this stage
produced is new.

The one file that changed and changed back is the main pLCA results table, which
a smoke run overwrote and which was restored from the previous commit and
verified at 60,001 lines. See the open item above.

---

## 7. Inputs and outputs

**Read.** The frozen raw empirical extract and its record metadata; the synthetic
corpus; the existing pLCA results table, for the gap analysis only; and the
published KL2 paper, for the two A_IQR construction details.

**Written.** Three source modules and their two test files; five audit scripts;
new sections in notebooks 1 and 3; thirteen new result tables; two new figures,
the calibration curve and the map of which categories can safely assume uniform
weights;
fourteen new decisions in the project brief's decision log, numbered 91 through 104,
and its handoff specification; manuscript discrepancy entries 81 to 95; and this
file.

**Not touched.** The generator, the synthetic corpus, the empirical extract, the
fitting methods, the scoring criterion, and the manuscript.

---

## 8. Next stage

**Stage 2e**, the pLCA construction. Its first job is the sweep over the number
of materials per pLCA, because the flip thresholds in section 4.4 are conditional
on four materials of equal use intensity and will move when that changes.

Common random numbers are worth installing while it is in there -- a tested
implementation is in `src/flip.py`, and the next Claude Code session should read
that before writing another -- but they are a refinement rather than a repair.
Every continuous output already separates the six methods far more than sampling
variation does.

**The second thing 2e should do is the experiment Stage 2c identified and nobody
has run**: the probabilistic LCA against the TRUE parent distributions, on the
same common random numbers, reporting how far each method's answer sits from the
truth rather than how far its fitted curve sits from a target. It may show that
the differences do not matter, which would be the cleanest result this paper
could report.

**One thing 2e should NOT do** is re-derive the flip threshold. It is calibrated,
it is recorded as a constant in the code with its provenance, and notebook 3
prints the recomputed crossings beside the stored ones on every run so that drift
is visible.
