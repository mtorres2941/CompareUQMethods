# HANDOFF stage-2f - The metric reduction

US spelling throughout, as in every file this project writes.

**HOW TO READ THIS FILE. Its reader does NOT have this repository** -- no
CLAUDE.md, no other report, no table, no figure, no source. Every claim below
is therefore stated in full where it is made, and a trailing `decision N` or
`entry N` is a citation into the project's decision log or its manuscript
discrepancy log, never the substance of the sentence. Where a file has to be
opened, the instruction is addressed to the NEXT CLAUDE CODE SESSION and says
so. The two figures are embedded and their captions carry the numbers, so a
reader whose copy does not render the images loses nothing.

# IF YOU READ ONE PAGE, READ THIS ONE

**Stage 2f asks which statistical characteristics of a material's declaration
set actually tell you which uncertainty method to use, and turns the answer
into a rule somebody can follow.** Each claim carries the number behind it and
a plain-language **so what** for a reader who builds buildings rather than
statistical models.

**THIS PAGE HAS BEEN REWRITTEN TWICE AND BOTH EARLIER VERSIONS WERE WRONG IN
THE SAME WAY: they ranked characteristics using a statistic that rewards
adding terms, computed on 127 real material categories.** What survives that
correction is below. Two specific claims are withdrawn and are named in claim 8,
because a reader who saw the earlier version needs to know which sentences to
stop repeating.

## 1. The rule: use a kernel density estimate above about 100 declarations

Above roughly 100 environmental product declarations in a category, fit a
kernel density estimate, and use market-share weights if you know the market
shares. Below it, fit a three-parameter lognormal. Measured against the best
choice that could possibly be made for each dataset individually -- a standard
nobody can reach, because it requires knowing the answer first -- that rule
costs **38.4 percent** more error at its best setting, against **53.6 percent**
for always using a kernel estimate and **178.5 percent** for always using a
lognormal.

**The cutoff is a basin and not a point, and the paper must say so.** The
lowest cost is at 75 declarations, and every threshold from **59 to 134** is
just as good. The worst case halves across the same region: a category where
the rule goes badly wrong costs 26.6 times the best possible below a threshold
of 53, and 13.1 times above it.

> **So what.** If a material category in your model has more than about a
> hundred declarations behind it, the flexible method is worth using and it is
> worth paying attention to which products actually sell. Below that, a simple
> skewed curve fits better, because there is not enough data to learn a shape
> from. The exact number is not delicate -- anywhere between 60 and 130 works
> identically -- so "about a hundred" is an honest way to state it and a
> precise-sounding number would be false precision.

## 2. No single method is best regardless, and the leader changes twice

Share of the 10,000 synthetic datasets on which each method comes closest to
the true distribution the data were drawn from:

    declarations   KDE      KDE      Lognormal  Lognormal  Normal   Normal
                   market   equal    equal      market     equal    market
    3 to 9          22.5     33.8      20.9        9.9       7.6      5.4
    10 to 99        20.0     22.5      25.2       17.9       8.8      5.6
    100 to 999      42.8     28.8       8.8       16.8       1.3      1.4
    1000 and up     69.6     23.5       0.6        6.1       0.0      0.2

**Normal distributions are never the answer**, at 0.0 to 8.8 percent in every
band, which is the strongest negative result the study has.

> **So what.** There is no method you can adopt once and stop thinking about.
> But there is one you can stop using: fitting a normal curve to embodied
> carbon data is the worst choice at every dataset size, and it is what most
> practice does today.

## 3. The advantage is U-shaped, and the dip is real rather than noise

A kernel estimate is closer to the truth on **62.6 percent** of datasets with
three to nine declarations, falls below half between about 10 and 55, and then
climbs to **86.5 percent** under equal weights and **96.8 percent** under
market-share weights at the top of the range.

The dip has a mechanism. With three to nine values there is no shape to
estimate and both families do equally badly, so the flexible one is nominally
ahead on a coin flip. Between ten and fifty, the lognormal's built-in shape
assumption is worth more than the kernel's freedom to follow the data. Above
that the data outweigh the assumption. **The figure draws the dip rather than
smoothing it**, because a clean monotone curve there would be the conclusion
and not the measurement.

> **So what.** The advice is not "more data is always better for the flexible
> method". There is a genuinely awkward middle -- roughly ten to fifty
> declarations -- where a simple assumed shape beats trying to learn one, and
> that is where a great many real material categories sit.

## 4. Nothing except dataset size gives a usable threshold, multimodality least

Every candidate characteristic was swept for a value at which the kernel
estimate overtakes the lognormal. Only dataset size produces one.
**Multimodality runs the wrong way**: as the modality index rises, the kernel
estimate's win share falls from 60.9 to 55.9 percent under equal weights, and
under market-share weights it crosses downward, 67.0 to 46.5 percent. Holding
dataset size fixed does not rescue it. Silverman's critical bandwidth is flat,
77.5 to 76.0 percent.

This confirms an earlier finding of this project on far stronger evidence:
10,000 datasets measured out of sample, where the earlier version had 127
measured in sample.

> **So what.** The intuition that lumpy, multi-peaked data is where a flexible
> method earns its keep is wrong, and it is wrong in the interesting direction.
> A kernel estimate's advantage comes from matching the general shape of a
> distribution -- its skew and its tail -- not from resolving separate humps.
> Counting peaks in your data will not tell you which method to use.

## 5. The one large non-size effect is a property of the weights, not the data

The biggest single effect anywhere in this analysis is the amount by which
market-share weighting shifts a category's average. It changes the kernel
estimate's win share from 31.5 to 67.5 percent under equal weights, and from
81.5 to 34.7 percent under market-share weights, as it goes from below to above
its crossing value.

**It cannot be used in a rule**, because computing it requires already knowing
the market shares, which is exactly what a practitioner does not have. It is
noted and set aside.

> **So what.** The single most informative thing about a material category is
> not a property of the numbers at all -- it is how different the market-weighted
> average is from the plain average. Nobody can compute that without production
> volumes that are not published, which is the strongest argument this study
> produces for publishing them.

## 6. Market-share weighting pays when shares are CONCENTRATED, not when even

Inside a single size band, splitting by how concentrated the weight vector is,
the share of datasets where the market-share fit beats its own equal-weighted
twin:

    weights spread over...        kernel estimate   lognormal
    the fewest products  (0.23)        74.4            79.4
                         (0.33)        70.7            76.0
                         (0.42)        61.6            66.1
    the most products    (0.51)        29.8            37.9

**This inverts the usual intuition.** Concentration is normally read as a
shrunken sample and therefore a cost. That is only the variance half. A nearly
even weight vector carries no information about the market, so the
market-share fit is the equal-weighted fit plus noise and it loses.

At three to nine declarations the same split is flat -- 41.4, 36.5, 36.3, 41.9
percent -- and market-share weighting loses regardless, on 39.0 percent of
datasets with an interval of 37.2 to 40.9. The reason is a different one: a
flat spread of market shares over three to nine points leaves a **median
effective sample size of 2.7**, with 93.1 percent of those categories below
five effective observations. Nothing about the weights rescues that, because
the problem is the point count.

> **So what.** Market-share weighting is worth the trouble when a few products
> dominate the market, which is the normal situation in construction materials
> -- one published figure has a single steel route at 64 percent of world
> production. If the market is genuinely fragmented and every product has a
> similar share, weighting adds noise and no information. And below about ten
> declarations, do not bother at all.

## 7. This could only be measured on the synthetic data, and that is the finding
## that justifies having built it

**The 127 real material categories cannot answer this question, and the way
that shows up is worth stating precisely.** The baseline model -- dataset size
and dispersion, nothing else -- was refitted 20 times, changing only which
categories fall into which cross-validation fold:

    arm                  datasets   median R2    range across reshuffles
    synthetic, equal       10,000     +0.280     +0.279 to +0.280
    synthetic, market      10,000     +0.524     +0.523 to +0.525
    real, equal               127     +0.326     -0.442 to +0.370
    real, market              127     -0.952     -2.605 to +0.337

**The corpus reproduces its own answer to three decimal places on every
reshuffle. The real arm does not reproduce its own sign.** A negative value
means the model predicts worse than simply guessing the average. So any single
number quoted from the real categories -- including a flattering one -- is one
draw from a three-point-wide distribution rather than a measurement.

On the synthetic arm the effects are real and small: under market-share weights
17 of 23 candidates clear their own noise, because that noise is only 0.003 on
10,000 datasets, but the largest effect is 0.034 against a baseline of 0.525.

> **So what.** This is what the synthetic datasets are for. There are only
> about 150 real material categories with enough declarations to analyze, and
> that is too few to answer "which method should I use, and when" with any
> stability at all. Generating ten thousand datasets whose statistical
> character matches the real ones is what makes the answer hold up.

## 8. Two claims from the earlier version of this stage are withdrawn

**The author's modality index is NOT the second-best predictor of which method
to use.** That claim came from an in-sample statistic on 127 categories.
Measured out of sample on 10,000, the index adds 0.0015 and 0.0027 -- not
distinguishable from zero -- while the visible mode COUNT, which the same
earlier version called the worst of all candidates, is the modality measure
that survives selection. **What does stand is the underlying defect it found**:
the index had been computed at a smoothing bandwidth the study abandoned four
stages earlier, and correcting that was right.

**The corpus is NOT the weaker arm for this question.** The earlier version said
its effects were an order of magnitude smaller than the real arm's because it
fails to span the real range. Both halves are wrong. The real arm's larger
numbers were overfitting, and the range shortfall is a **single category** --
`Aggregates`, a contaminated database bin with a coefficient of variation of
6.93 where the rest of the arm reaches 2.4. Removing every real category above
the corpus's range moves the kernel estimate's win share from 40.2 to 40.5
percent and the size crossover from 124 to 122 declarations.

> **So what.** The synthetic data covers what real material categories look
> like. The one thing it does not reach is a database category that is not a
> material at all -- a bin holding sinks, worktops and gravel together -- and
> that is a statement about the database's filing rather than a limit on the
> conclusions.

## 9. How significance is decided here, since it is not by p-value

No p-value appears in any figure, table or claim of this stage. Each
characteristic's contribution is measured as the improvement in prediction on
data the model has not seen, reported beside the spread of that improvement
across folds. **An effect smaller than its own spread is not reported as an
effect**, however small its p-value would have been.

Redundancy is handled by selection rather than by discarding correlated
columns. Skewness, kurtosis and the two normality statistics are four views of
one underlying shape, so a table that tests each one alone credits the same
effect four times. A characteristic enters the reported set only if it still
improves prediction once everything already chosen is in the model.

> **So what.** The usual way of deciding what matters -- a significance test on
> each characteristic in turn -- would have produced a longer list, counted the
> same effect several times over, and rewarded the analysis for adding terms.
> What is reported instead is how much each characteristic actually improves a
> prediction of data it has not seen.

## 10. The synthetic data was NOT regenerated, and that was tested rather than assumed

Widening the corpus's dispersion was investigated at the author's instruction
and is not adopted. The generator can reach further than an earlier stage
concluded -- that stage's audit never swept the parameter that actually binds,
the truncation width -- but the price is the quantity the paper is about: the
best widening halves the dispersion mismatch and **multiplies the
uniform-to-market-share distance mismatch by 2.4**, while worsening the overall
calibration by 6.3 times its own seed-to-seed noise. It still does not reach
the real maximum. Since the gap is one contaminated category and removing it
changes nothing (claim 8), generation stays closed.

Dataset size was also left alone: the choice curve flattens to a slope of
-0.233 per tenfold increase above 1,000 declarations, the kernel estimate
already wins 96 percent there, and the three real categories larger than the
corpus's maximum behave like the band below them.

> **So what.** Nothing downstream was invalidated and no number in the paper
> moved. The limitation to state is narrow and specific, rather than a general
> caveat about synthetic data.

## The two figures

![Which method, and when](../outputs/figures/CompareUQMethods_FIG_WhenToUseWhich.png)

**Figure A. Left:** the share of datasets on which a kernel density estimate is
closer to the truth than a three-parameter lognormal, against the number of
declarations, with a bootstrap band. It starts at 62.6 percent at three to nine
declarations, dips below half between about 10 and 55, and rises to 86.5
percent under equal weights and 96.8 percent under market-share weights. The
grey band marks 59 to 134 declarations. **Middle:** the cost of a
"kernel estimate above the cutoff, lognormal below" rule against the best
choice that could be made per dataset, as the cutoff moves. The minimum is 38.4
percent at 75 declarations and the curve is flat from 59 to 134, which is what
licenses quoting a round hundred. Always using a kernel estimate costs 53.6
percent; always using a lognormal costs 178.5 percent. **Right:** which of the
six methods is actually closest, by size band. The kernel estimate with equal
weights leads below ten declarations, the lognormal with equal weights from ten
to ninety-nine, and the kernel estimate with market-share weights above a
hundred, reaching 69.6 percent.

![What else changes the answer](../outputs/figures/CompareUQMethods_FIG_ChoiceDrivers.png)

**Figure B.** How much each characteristic improves a prediction of which
method fits better, measured on data the model has not seen, once dataset size
and dispersion are already accounted for. Bars in orange are the ones kept once
everything already chosen is in the model; the whisker is the spread across
folds, and a bar shorter than its whisker is not an effect. **Left, equal
weights:** one characteristic stands out, the distance between the
equal-weighted and market-share-weighted versions of the same dataset, at
+0.170; everything else is at or below 0.003. **Right, market-share weights:**
the shift in the average leads at +0.034, then the weight carried by outliers
at +0.026 and the visible mode count at +0.019. **All of these are small**
against baselines of 0.280 and 0.525, which is the point: once you know how
many declarations there are and how spread out they are, little else changes
the answer.

## 1. Stage and branch

| | |
|---|---|
| **Stage** | 2f, the metric reduction |
| **Branch** | `stage-2f-multivariate` |
| **Branched from** | `1e8c884` on branch `stage-2e-plca`, working tree clean |

Twenty-two commits, in logical units so any moved number can be bisected. The
number-moving ones are named in section 6.

---

## 2. What was asked

Resolve the Shapiro-Wilk versus Shapiro-Francia inconsistency and fix the
Royston p-value. Then replace the rolling-average presentation of
goodness-of-fit against the statistical characteristics -- 21 panels per arm,
no uncertainty band, no data density, and a set of characteristics correlated
with each other so that the marginal views overstate how many independent
effects exist. Fit a multivariate model of the fit score and, separately, of
which method wins, on every characteristic at once, using an interpretable
model alongside a flexible one with permutation importance and partial
dependence; rank the characteristics by independent contribution and say which
small subset carries the signal. Run the reduction TWICE, once against the fit
score and once against the error in the ANSWER, and report what survives one
and not the other. Treat dataset size as a first-class predictor and a
confound; handle the characteristics that are undefined in the smallest size
stratum explicitly; report every aggregate at equal allocation and reweighted
to the real size mix; treat the three modality measures as three candidates;
and say whether the characteristics that predict are the ones where the
synthetic corpus is densest, which would narrow the generalization claim.

---

## 3. What was done

**One new source module with 30 tests**, holding the candidate set and its
modeling scales, the missingness and rows-used reports, the size-confound
measurement, two model families ranked by permutation importance on held-out
folds, the tautology guard, partial dependence, the empirical size-mix
resample, the three-way modality comparison, and the binned and smoothed curves
that replace the rolling averages. Sixteen new cells at the end of the third
notebook are each one call into it.

**IT IS IN THE THIRD NOTEBOOK AND NOT THE SECOND, deliberately.** The reduction
is run against two kinds of target and the second -- the error in the answer --
exists only after the probabilistic LCA has been run against the true
distributions. Putting the reduction in the second notebook would make it read
a table the third one writes and break the run order from a clean checkout.

**The whole test suite is 487 and all pass**, including the eight regression
fixtures that pin the characteristics and all six goodness-of-fit scores.

**Five defects were found by running the work**, each fixed with a test: a
crash that killed the first full reduction part-way through, a unimodal share
divided by the wrong denominator, a cached file that failed the second notebook
nineteen minutes into a run, a comparison made in two different units, and a
figure cell that would have run for hours. They are described in section 5.

---

## 4. What the measurements say

### 4.1 The survivors

Pooled over both target families, both arms, all six methods and both model
families -- 96 models -- ranked by mean permutation-importance rank on the
held-out fold, with the definitional candidate of section 4.2 removed:

| characteristic | mean rank | share of models it is top-five in |
|---|---|---|
| coefficient of variation, variable | **3.09** | 84 pct |
| coefficient of variation, uniform | **3.80** | 78 pct |
| entropy, variable | **3.93** | 83 pct |
| dataset size | **5.06** | 70 pct |
| entropy, uniform | **5.24** | 70 pct |
| the dataset mean | 6.79 | 49 pct |
| ... | | |
| Silverman's critical bandwidth | 13.2 | **0 pct** |
| visible mode count | 19.5 | **0 pct** |

The five reduce to two quantities: entropy is dataset size (spline R2 of 0.924
on the corpus, 0.839 on the real arm) and the uniform coefficient of variation
is the variable one (correlation 0.979). Decision 129.

**THE SET IS STABLE AND THE ORDER IS NOT.** A separate run on an independent
random stream returns the same five, with mean ranks agreeing to 0.03 to 0.25,
and the second and third places swap. The paper states a SET.

### 4.2 A characteristic that IS part of the score

Every model in this study is scored against the market-share-weighted empirical
distribution of the data, including the three equal-weighted fits, so an
equal-weighted model carries a distance no estimation method can remove. That
distance is exactly the uniform-to-variable Wasserstein distance, which the
study reports as one of its characteristics.

**Its rank correlation with that irremovable part is 1.000000 for all three
equal-weighted methods on both arms.** On that identity alone it reaches a
correlation of **0.966** with the in-sample score of the kernel estimate. Any
sentence of the form "the fit gets worse as the uniform-to-variable distance
grows" is, for an equal-weighted method, a restatement of the definition.

**Its standing on the downstream error is real**: the market-weighted methods
have an irremovable part of exactly zero, and it still correlates **0.72, 0.72
and 0.57** with the error in a material's estimated contribution. So every
survivor ranking is reported with and without it. Decision 130, entry 118.

### 4.3 The two targets

| characteristic | rank on the FIT | rank on the ANSWER | shift |
|---|---|---|---|
| dataset size | 6.90 | **4.42** | **-2.48** |
| skewness, uniform | 15.02 | 12.65 | -2.38 |
| kurtosis, uniform | 14.35 | 12.04 | -2.31 |
| entropy, uniform | 7.13 | 4.90 | -2.23 |
| coefficient of variation, uniform | 3.38 | 5.40 | +2.02 |
| lognormal fit statistic | 14.13 | 16.35 | +2.23 |
| normal fit statistic | 9.88 | **13.85** | **+3.98** |
| weight of outliers | 13.25 | **17.29** | **+4.04** |

Size and its proxy rise when the target becomes the answer; the shape
statistics fall. Decision 131.

**And the error in a material's rank-1 frequency is 0.085 to 0.096 predictable
from its dataset's characteristics**, against 0.62 to 0.66 for the error in its
contribution. A rank-1 frequency is a property of the group of four materials,
not of the dataset, so the dataset's own characteristics cannot carry it.

### 4.4 Which method wins IS predictable, and size is what predicts it

Asked within a weighting scheme, which is the only valid form of the question,
with the majority-class baseline beside every accuracy because on a question one
method wins 70 percent of the time an accuracy of 0.70 has learned nothing:

| target | arm | weighting | accuracy | baseline | lift |
|---|---|---|---|---|---|
| cross-validated | real | equal | **0.827** | 0.504 | **+0.323** |
| cross-validated | real | market | 0.771 | 0.575 | +0.197 |
| against the known parent | corpus | equal | 0.695 | 0.566 | +0.129 |
| against the known parent | corpus | market | 0.738 | 0.644 | +0.094 |

**The strongest single predictor of which method wins is dataset size**, at a
mean importance of 0.084 against 0.051 for the next. That CONFIRMS what an
earlier stage found by a completely different route -- regressing one method's
score against another's on the logarithm of the count -- and the confirmation is
worth stating because the two computations share nothing but the data.

### 4.5 Modality

| measure | adds over size alone, real arm | corpus | significant on |
|---|---|---|---|
| Silverman's critical bandwidth, variable | **0.210** | **0.175** | every model |
| Silverman's critical bandwidth, uniform | 0.205 | 0.171 | every model |
| the continuous modality index, variable | 0.128 | 0.042 | every model |
| visible modes at the fitted bandwidth | 0.012 | 0.013 | **none of 12 on the real arm** |
| visible modes at Scott's bandwidth | 0.033 | 0.004 | 9 of 12 |

And in the full model the critical bandwidth is 14th of 22 and top-five in zero
of 96 models, correlating +0.54 and +0.60 with the coefficient of variation.
Decision 132, entry 119.

### 4.6 What the models were told about missing data

**Two different things remove the smallest datasets and neither announces
itself.** Excess kurtosis is undefined below four values -- 6 of the 20 real
categories with 3 to 9 EPDs, and 612 of 2,500 synthetic ones -- and a visible
mode count needs eight, which removes 17 of 20 and 2,026 of 2,500. Every other
size band is complete on every characteristic. **A model that dropped incomplete
rows would discard 85 percent of the real 3-to-9 band and 81 percent of the
synthetic one** and leave the rest untouched, which is precisely the regime
where a parametric family is expected to beat a kernel estimate. Both models
keep every row instead: the interpretable one fills a missing value and adds a
column saying it was missing, the flexible one splits on missingness directly.

**And the cross-validated target on the real arm is undefined below ten
values**, because half of a nine-value dataset is four values. That reduction
uses 127 of 147 categories and **0 of the 20** in the smallest band. Entry 122.

### 4.7 Post-stratification

Reweighting the corpus to the real size mix -- 13.6, 53.1, 25.9 and 7.5 percent
across the four bands, giving 4,712 datasets -- roughly halves the importance of
dataset size, **0.134 to 0.064**, and of entropy, 0.110 to 0.075, while the
coefficient of variation barely moves, 0.270 to 0.253. **The survivor set and
its ordering do not change.** So the reduction is robust to the allocation, and
under the real size mix dispersion dominates size by more than equal allocation
suggests. Both are reported, as every headline in this study is.

### 4.8 Coverage against importance

Median margin beyond the real maximum: **-0.629 for the five survivors and
+0.509 for the other seventeen**; rank correlation between importance and margin
**+0.484**. The corpus does not reach the real maximum on the coefficient of
variation under either weighting or on dataset size, and has five to seven real
ranges of headroom on skewness, kurtosis and the modality index.

This sharpens the existing coverage limitation rather than reversing it: that
limitation was argued on WHICH categories are uncovered, and this adds that the
shortfall sits on the most predictive characteristic in the study. **Generation
stays closed.** Decision 133, entry 120.

---

## 5. What is still open

### Owned by a later stage

| Item | Owner |
|---|---|
| The `(1-capecc)` divisor, partly resolved in the previous stage and still to be reviewed; magnitude-based companion metrics; the sensitivity of the headline rank-1 frequency. **This stage adds a third independent argument for demoting the ranking metrics: the error in a rank-1 frequency is only 9 percent predictable from a dataset's own characteristics, because it is a property of the group** | 2g |
| The profile-likelihood guard sweep; the Dirichlet concentration sweep, which should vary the block structure and not only the parameter; multiple weight realizations; the deduplicated variant; the mode-share coupling; **and the pedigree matrix**, which is what connects this paper to the practice most readers use | 2h |
| Every figure brought to the style guide. **This stage found that the guide's own clash detector had never checked a single panel title** and fixed it, so a session doing that work now has a tool that works. The older figures still carry a Unicode minus | 3 |
| **The reduced figure is what the manuscript's metric count should describe.** The full candidate set belongs in the supplement and the survivors in the main text | manuscript |
| A real-building anchor, if citing the staircase paper is not enough | 2i, optional |
| An industry-average EPD as a direct estimate of the market-weighted mean | unowned |

### Known and accepted

**The survivor ORDER is not stable across random streams** and the set is; the
paper must state a set. **The out-of-sample reduction on the real arm excludes
the 20 smallest categories** and those claims rest on the in-sample score.
**Partial dependence splits an effect between two strongly correlated
characteristics** rather than assigning it to one, so a moderate partial
dependence is not evidence of an independent effect; read it beside the
correlation table. **The empirical Silverman multimodality share carries a few
points of estimator noise** -- 49.3 percent at 100 bootstrap replicates against
45.6 at 60 -- and the table that reports it is computed at 60 and now carries
its replicate count as a column, so no figure drawn from it can quote a share
without its provenance.

**One item for the NEXT CLAUDE CODE SESSION, not for the manuscript reader:**
the two derived replay caches are now excluded from version control, this
stage's own pair having been removed from it; the pair belonging to the
previous corpus is still tracked and removing it belongs to the deposit stage.

---

## 6. Numbers that moved

**Three changes moved a committed number and no fit, score or probabilistic LCA
result is among them.**

**The normality statistic.** Only the two EQUAL-WEIGHTED columns; the
market-weighted ones were already Shapiro-Francia and are bit-identical on both
arms. Median absolute change on the real categories: **0.0103** at 3 to 9 EPDs,
0.0062 at 10 to 99, 0.0029 at 100 to 999, 0.00005 above 1,000; arm mean 0.7862
to 0.7834. Largest single change on either arm 0.0358. On the corpus, 9,388 and
9,387 of 10,000 rows move.

**Everything the statistic does NOT reach, checked rather than assumed.** No
goodness-of-fit score moves on either arm. The 60,000-row probabilistic LCA
results table is identical column for column. Nine previously committed
compressed tables -- the run against the truth, the design comparison, the
oracle weights, the flip calibration -- are byte-identical. The generator's
calibration is unchanged to the last digit, because the objective it is tuned on
reads only market-weighted columns. The corpus was RECOMPUTED and not
regenerated: its values, its parent record, its groupings and both replay caches
are byte-identical to the corpus it came from, and only the two columns above
differ.

**The visible-mode counts** were computed on a 500-dataset sample of the corpus
and are now computed on all of it, which was under a minute of work. The
synthetic share with one visible mode moves by sampling error only; at the
bandwidth the study fits it reads **75.9 percent against a real-arm 68.5**, over
7,974 synthetic datasets and 130 real categories with at least eight values.

**The smoothed curves** gained a bound: restricting the smooth to the span the
binned summary covers removes every fitted value at or below zero, **24 of
15,000 rows**, and the minimum fitted distance goes from **-0.0119 to +0.0065**.
A negative Wasserstein distance is not a quantity. The binned rows, which carry
the uncertainty and the counts, are identical.

**The three regression fixtures were re-frozen in the commit that moved them**,
with the delta recorded there and in the fixture README, and all eight
regression tests pass.

### The five defects, each now with a test

1. **A crash killed the first full reduction part-way through.** The function
   that measures what a characteristic adds over dataset size took size from
   the candidate list rather than from the data, so calling it with a list that
   does not name size -- which the three-way modality comparison does -- asked
   for a column the caller never supplied.
2. **The unimodal share was divided by the whole arm** rather than by the
   datasets the measure is defined on, counting an undefined dataset as
   multimodal. It read 0.605 on the real arm against a true 0.685.
3. **A cached file failed the second notebook nineteen minutes into a run.**
   The recompute path copies a replay cache that records which corpus it was
   built for, and the loader correctly refused it. The guard stays and is
   tested; the copy is now relabelled with its origin recorded.
4. **A comparison was made in two different units.** The partial dependence is
   fitted on the logarithm of the target and the marginal curve was drawn on
   the target's own scale, so the ratio of the two was meaningless and returned
   "fractions surviving" of 2.5 to 7.9.
5. **A figure cell would have run for hours.** Bootstrapping the smoother 400
   times over five characteristics, six methods and three targets is some
   12,000 fits at 600 ms each. Turning off the smoother's robustifying
   iterations is 255 times faster and is NOT available -- measured, it moves the
   curve by 54 percent of its own range, six times the width of the band it
   would be drawn inside -- so the uncertainty now comes from the binned
   bootstrap, which is exact and cheap, and the smoother is fitted once. The
   cell takes seventeen seconds.

**And one defect in the tooling.** The clash detector written two stages ago to
stop figure labels colliding had never checked a single panel TITLE: the
plotting library keeps a separate text object for a left-aligned title, the
detector read the centred one, and this project sets every title left-aligned.
It reported no overlap on a five-column figure whose titles plainly overlapped.

---

## 7. Inputs and outputs

**Read.** The frozen raw empirical extract; the synthetic corpus and the
parents and mode labels recovered from it; the goodness-of-fit and
cross-validated scores; and the previous stage's run of the probabilistic LCA
against the true distributions, which is what makes the second target exist.

**Written.** One source module and its test file; one more test file for the
corpus recompute; one audit script; a fourth notebook holding the whole
reduction; result tables and figures as listed below; fifteen decisions in the
project's decision log, numbered 125 through 140; manuscript discrepancy entries
114 through 129, with entries 9 and 11 marked resolved; a rewritten README; and
this file.

**Added by the author's review**, after the first version of this stage was
found to have ranked characteristics on an in-sample statistic computed on 127
real categories: six tables named `TABLE_Reduction*` covering the
cross-validated gains, the forward selection, the baseline stability across
fold assignments, the best method by size band, the policy cost curve, and the
weight-concentration split; two rebuilt figures; and twelve tests.

**Not touched.** The generator, the corpus's values, the empirical extract, the
fitting methods, the scoring criterion, the published flip thresholds, the
probabilistic LCA, and the manuscript.

---

## 8. Next stage

**FIRST, WHAT THE NEXT SESSION MUST NOT REPEAT.** Three claims from this
stage's first version are withdrawn and are recorded as withdrawn in the
project's decision log at entries 134, 135 and 136: that the author's modality
index is the second-best predictor of which method to use (it is an in-sample
result on 127 categories and does not survive out of sample), that five further
characteristics add 0.10 to 0.14 of explained variance to the choice (the same
defect), and that the corpus is the weaker arm for this question because of its
range (it is not; the shortfall is one contaminated category). **Any ranking of
characteristics must be cross-validated and must be taken from the synthetic
arm**, because the real arm's baseline does not reproduce its own sign across
fold assignments.

**Stage 2g**, the metric set: the sensitivity of the headline rank-1 frequency,
magnitude-based companions, and the `(1-capecc)` divisor.

**What this stage hands it.** A third independent argument for demoting the
ranking metrics, and this one is about predictability rather than fragility: the
error in a material's chance of leading is only **9 percent** predictable from
its dataset's own characteristics, against **62 to 66 percent** for the error in
its estimated contribution, because a rank-1 frequency is a property of the
group of materials and not of the dataset. The previous stage recommended
reporting five kinds of statement with the uncertainty index added; nothing here
contradicts that and this strengthens the case for leading with magnitudes.

**And what it should NOT redo.** The reduction is done and the survivor set is
stable. If a later stage wants a characteristic back in the figure, the question
to ask is whether it earns a place once dispersion and dataset size are already
there, which is what the partial-dependence table answers.

### Habits, added by this stage

12. **Ask whether a predictor is part of the thing it predicts.** The strongest
    apparent result in the first reduction was an identity with a correlation of
    exactly 1.000000.
13. **Look at the figure.** Three defects survived every table and every test
    and were visible in one glance at the rendered image, including one in the
    tool whose job was to catch them.
14. **A speed-up that changes the answer is not a speed-up.** Measure what the
    fast path costs before taking it; here it was 255 times faster and moved the
    curve by six times the width of its own uncertainty band.
15. **An in-sample fit statistic is not a measurement, and on a small sample it
    is not even close.** Adding five spline terms to 127 observations raises
    in-sample R2 by about 0.043 for free, which was most of what this stage
    first reported as signal. Everything is now scored on data the model has not
    seen.
16. **Before quoting a number, refit it on a different fold assignment.** The
    same model on the same data returned a baseline R2 of -0.724 and -2.441 on
    two runs differing only in a random seed. Reporting either would have been
    reporting a draw. The instability was the real finding and it is what
    settled which arm the stage rests on.
17. **A ranking of one-at-a-time tests counts the same effect several times.**
    Skewness, kurtosis and two normality statistics are four views of one shape.
    Selection against everything already chosen is the fix; pruning correlated
    columns by a threshold is not, because it throws away the choice of which
    view to keep.
18. **When a helper silently changes a figure, fix the helper.** `finish` was
    thinning tick locators on categorical axes, so a panel came back with two of
    four band names showing and no error anywhere. A test now pins it.
