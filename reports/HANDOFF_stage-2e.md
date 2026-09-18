# HANDOFF stage-2e - The pLCA construction

US spelling throughout, as in every file this project writes.

**HOW TO READ THIS FILE. Its reader does NOT have this repository** -- no
CLAUDE.md, no other report, no table, no source. Every claim below is therefore
stated in full where it is made, and a trailing `decision N` or `entry N` is a
citation into the project's decision log or its manuscript discrepancy log,
never the substance of the sentence. Where a file has to be opened, the
instruction is addressed to the NEXT CLAUDE CODE SESSION and says so. The two
figures are embedded, and their captions carry the numbers, so a reader whose
copy does not render the images loses nothing.

# IF YOU READ ONE PAGE, READ THIS ONE

**Stage 2e asks what the choice of UQ method does to a probabilistic LCA, and
it is organised around the five statements such a study makes.** Each claim
below is one the paper can make, with the number behind it. The order is the
order the results section should take.

## The decision a designer actually makes is not sensitive to the choice

1. **Two designs differing in one material: the choice of UQ method changes the
   stated probability that the substitution is an improvement by AT MOST TWO
   PERCENTAGE POINTS, and every method is within two and a half points of the
   truth.** Over 800 option pairs scored against the true distributions, with a
   substitution claimed to save 0, 1, 2, 5, 10 and 20 percent of the building,
   the true probability that B beats A is **0.505, 0.529, 0.554, 0.629, 0.754
   and 0.953**, and the spread across the six methods is **0.006, 0.006, 0.007,
   0.012, 0.020 and 0.012**. This is the cleanest result the stage produced and
   it is a null: on the comparison a designer makes, the choice is safe.
2. **A material must lead the next by about a factor of two before the choice
   of method cannot change which one leads.** The probability that it changes
   the leading material crosses 1 percent at a top-two contribution ratio of
   **2.13** (95 percent interval 2.09 to 2.17), 5 percent at 1.64 and 10
   percent at 1.46. **The one real building element available sits at 1.02**:
   Marsh, Lewis, Hattam and Allen (in press) report their Concrete-Precast
   staircase's top two products at 42 and 41 percent under the ICE recommended
   factors.

## Where the choice does matter

3. **Every method understates the building's upper tail**, its 90th percentile
   by 0.09 to 0.36 on a four-material building whose total averages 4.0, and
   they disagree about the budget statement in both directions: at a budget the
   truth meets 90.0 percent of the time, `Lognormal, Variable` reports **91.3**
   percent and `Normal, Uniform` reports **86.8**.
4. **The value of a specification policy is where the normal fails hardest.**
   Against a true mean saving of **5.39 percent** of the building from capping
   a material at the 75th percentile of the declarations held, the normal
   reports 6.19 and the lognormal 4.88; asked for the chance of achieving at
   least a 5 percent saving, **the truth is 23.2 percent and the normal says
   30.5**, an overstatement of 7.4 points, while the kernel estimate is within
   1.3 and the lognormal within 0.3.
5. **A quantity reduction is method-independent to four decimal places**, and
   the contrast explains the whole stage: using 25 percent less of a material
   is a deterministic fraction of its own contribution, so no distributional
   assumption enters, while specifying a cap acts entirely through the upper
   tail, which is exactly what the methods disagree about.

## How the methods fail

6. **They fail in opposite directions on the average and the same direction on
   the tail.** A material's estimated contribution is biased **high** by the
   normal (+0.044 uniform, +0.050 variable), **low** by the lognormal (-0.038,
   -0.027), and almost not at all by the kernel estimate (-0.014, +0.003); its
   95th percentile is understated by all six, by 0.086 under the KDE and 0.193
   under the normal.
7. **They fail on the same materials, and the family is not what separates
   them.** Per-material errors correlate **0.892 to 0.970 between methods that
   share a weighting scheme** and only **0.581 to 0.714 across weighting
   schemes**, and all six err in the same direction on **51.2 percent** of
   materials against about 3 percent if they were independent. **Choosing a
   different family does not hedge the risk.**
8. **No method recovers the answer, and the spread matters more than the
   average.** The mean absolute error in a material's estimated contribution is
   0.12 to 0.17 where every material contributes 1.00, but the error on ANY ONE
   material has a standard deviation of **0.23 to 0.30** and a 99th percentile
   of **0.89 to 1.26** -- for one material in a hundred the method is wrong by
   more than the material's entire expected contribution.

## Weighting, and the machinery

9. **Knowing market shares buys about 17 percent; guessing them with a flat
   Dirichlet captures a third of that on the magnitude and none of it on the
   ranking.** Mean absolute error in a material's contribution: kernel estimate
   **0.1285** under uniform weights, **0.1215** under this study's drawn
   weights, **0.1064** under weights that know the true mode-level share;
   lognormal 0.1243, 0.1168, 0.1007. On the rank-1 frequency the last column
   gains as much again and the middle one gains nothing. **This is not a verdict
   on weighting**: the contrast is between knowing shares and guessing them, and
   what separates the oracle from the drawn weights is noise this generator
   introduces by construction and the real world does not have.
10. **Common random numbers were worth about one percent**, which is why
    installing them is a refinement and not a repair. The sampling noise is 4 to
    15 percent of the difference between two methods and adds nearly
    orthogonally, so no median moved by more than 2 percent. Where it mattered
    is an argmax: **3.67 percent** of comparisons named a different top
    contributor with no model difference at all, and that floor is now zero.
11. **The steadiest output is the one that sets data-collection priorities.**
    The uncertainty index has the lowest NRMSE between methods of any main
    output, **0.503** against **1.042** for a material's rank-1 frequency, and
    the study computes it and reports it nowhere. **It is also the fourth
    reduction strategy under another name**: three strategies reduce the
    expected impact -- use less, specify better, substitute -- and this one
    reduces the variance of the answer, which is what obtaining a
    supplier-specific declaration buys.

**What must travel with the ranking numbers, in the same paragraph and not a
footnote.** In the study's own pLCA every material carries a use intensity of
1.0, so the contributions are exchangeable and a ranking is as fragile as it
can be made; the sweeps in claims 1 and 2 vary that deliberately, and the
magnitude results do not depend on it. Material use intensity is also
deterministic here, while a real quantity take-off carries its own uncertainty,
which in practice can exceed the coefficient uncertainty this paper is about.

**Two housekeeping items, neither a finding.** A smoke run -- a reduced
execution used to check the pipeline runs end to end in a minute rather than
forty -- can no longer write into the results directory, which closes an item
the previous stage opened after one reached a commit and replaced the
60,000-row results table with a 960-row one. And three cells of the third
notebook had never run from a clean start, one of which draws a figure that is
in the deposit, so that figure could not be regenerated from the notebooks as
they stood.

---

## The two figures

![A dominant material protects the ranking, not the result](../outputs/figures/CompareUQMethods_FIG_MaterialDominance.png)

**Figure: a dominant material protects the ranking, not the result.** Each grey
point is one probabilistic LCA -- its own top-two contribution ratio against the
share of its fifteen method pairs that name a different leader -- with a rolling
mean through them. The upper panel falls from about 55 percent at equal
contributions to zero, crossing 1 percent at a ratio of **2.13**. The lower
panel is the change the same method pairs make to a material's estimated
contribution, and it is flat: **0.188 at equal intensities and 0.166 at
100:1**, an 11 percent decline while the ranking goes from a coin toss to
certainty. The comparison is in absolute units and that is legitimate because
the intensity vector is normalised to a mean of 1.0 in every case, so the
building's total mean contribution is the same number throughout.

**The lower panel is a rolling MEDIAN and the upper a rolling mean**, because a
flip rate is a mean of zeros and ones while the change in a contribution is
heavy tailed -- a few groups move by more than 1.4 where the typical one moves
by 0.19 -- so a mean there would track the tail rather than the typical group.
Reading a mean against a median is what made this panel appear to rise with
dominance when the median is flat: within four materials the median change runs
**0.187, 0.190, 0.198, 0.181, 0.182, 0.197** across ratio bands from 1.0 to
21.2.

**Both panels are held at four materials, which is the study's own
construction, and that is not presentation.** What a given top-two ratio implies
about dominance changes with the number of materials, and a Dirichlet draw over
twelve produces large ratios far more often than one over two, so pooling the
sizes fills the right of the axis with twelve-material groups whose
most-affected material moves more simply because a maximum over twelve is drawn
from more chances. Drawn that way the lower panel appears to RISE with
dominance, which is a group-size effect wearing a dominance label.

![The methods differ in which way they are wrong, not in how far](../outputs/figures/CompareUQMethods_FIG_PLCATruth.png)

**Figure: the methods differ in which way they are wrong, not in how far.** The
signed error of each method against the true distributions, for one material's
contribution and for the whole building's total, with the mean error marked.
The lognormals sit left of zero and the normals right of it while the kernel
estimates straddle it, and the widths are similar: the standard deviation of a
single material's error runs **0.234 to 0.301** across the six. An earlier
version of this figure plotted the mean with a confidence interval on that
mean, which over ten thousand materials is so narrow it read as though the
methods were nearly exact.

---

## 1. Stage and branch

| | |
|---|---|
| **Stage** | 2e, the pLCA construction |
| **Branch** | `stage-2e-plca` |
| **Branched from** | `80116bf` on branch `stage-2d-threshold`, which is a commit this stage made first: the Stage 2d session ended with its decision log, its discrepancy entries, its handoff and two figures uncommitted, and they are committed there unchanged |

Commits, in order: the pLCA engine and its tests; the smoke guard and a figure
that could not be drawn; common random numbers in the study's own pLCA plus the
sweep machinery; the notebook sections; a module-shadowing fix and the figures;
the mechanics documentation; and the results.

---

## 2. What was asked

Six things in the stage prompt, then a review that changed what the stage is
about.

**The prompt.** Sweep the materials per probabilistic LCA, because the flip
thresholds the previous stage published are conditional on four of equal use
intensity; sweep the intensities themselves on the simplex, reported against an
observable rather than a Dirichlet parameter, anchored to a real design if one
existed; make it impossible for a reduced test run to overwrite the project's
most expensive table; install common random numbers across the methods;
resample the groupings and put bootstrap intervals on every headline percentage
and NRMSE; and run the probabilistic LCA against the TRUE distributions the
synthetic data was drawn from.

**The review, which is where most of the value came from.** The first draft led
with the error in a material's chance of being the largest contributor, and the
author's objection was that a ranking statement is close to meaningless when
the top two are near-tied. What the stage was asked for instead was the set of
statements a probabilistic LCA actually makes, and the decisions a designer or
researcher takes from them. That produced: the five statements below; an ECC
cap that could measure what the strategy is for; a design comparison, which the
study had never made; a counterfactual that separates knowing market shares
from guessing them; a test of whether dispersion belongs in the safe-lead rule;
and both figures rebuilt.

## 3. What was done

**One new source module with 53 tests.** It holds the common-random-numbers
draw, the ten pLCA outputs in the notebook's own definitions, material use
intensity as a share vector, the resampled groupings, the cluster bootstrap,
NRMSE with an interval, the five statements, the design comparison and the run
against the true distributions. Every notebook cell below is one call into it.

**The third notebook gained eight sections and two rebuilt figures**, and its
main pLCA loop now uses common random numbers and an absolute specification
cap.

**The whole test suite is 439 tests and all pass**, including the eight
regression fixtures that pin the empirical metrics, the synthetic metrics and
all six goodness-of-fit scores. That is the check that the fitting, the corpus
and the empirical arm were not touched.

**Three defects were found by running the notebook end to end from a clean
start**, which had never been done for the cells the previous stage added: a
figure that used a name the notebook never defines, a call to a module function
when only the names imported from that module were in scope, and a loop index
that shared its name with the new source module so that every call into it read
an integer. All three are fixed and a test now refuses a notebook variable that
shadows an imported module.

## 4. What the measurements say

Organised as the results section should be: what the choice of method does to
the numbers, to where the uncertainty sits, and to the decision.

### 4.1 The five statements a probabilistic LCA makes

| | the statement | metric | where it stood |
|---|---|---|---|
| 1 | **Magnitude.** "This building is X, and I am this confident", including "what is my chance of meeting a budget" | the total's distribution and its quantiles | only a mean and a standard deviation were kept |
| 2 | **Attribution.** "This material dominates" | contribution, share, chance of leading | fully computed; what the paper reports |
| 3 | **Information.** "This material's uncertainty dominates, so measure it" | the uncertainty index | computed, reported nowhere |
| 4 | **Action.** "This intervention delivers X, and here is how likely" | the reduction's distribution | the means were kept, the confidence discarded |
| 5 | **Comparison.** "Option A beats option B" | discernibility, comparison index | never made |

**The fourth reduction strategy needed no code and that is the point.** Three
strategies reduce the expected impact -- use less of a material, specify a
better product, substitute a different one -- and a fourth reduces the VARIANCE
of the answer: obtaining a supplier-specific declaration, which is exactly what
the uncertainty index measures. Naming it that way is the contribution.

### 4.2 The comparison, which is the decision and is not sensitive

800 pairs of options sharing three materials and **the same random draws for
them**, differing in the fourth. Because every dataset here is normalised to a
mean of 1.0, a substitution alone changes the expected total by nothing, so the
replacement's use intensity carries a controlled expected saving -- realistic
in any case, since a substitute generally needs a different quantity.

| B is claimed to save | 0 pct | 1 pct | 2 pct | 5 pct | 10 pct | 20 pct |
|---|---|---|---|---|---|---|
| **the truth** | 0.505 | 0.529 | 0.554 | 0.629 | 0.754 | 0.953 |
| spread across six methods | 0.006 | 0.006 | 0.007 | 0.012 | 0.020 | 0.012 |

The normal is consistently the most optimistic and the kernel estimate the
least, the same ordering as everywhere else, but no design decision turns on a
gap of two points. **Every other comparison in this study is between a method
and another method, or between a method and a target it was fitted to. This one
is the decision, scored against the right answer.**

### 4.3 How far a material must lead before the ranking is safe

The sweep over group size and use intensity, 72 cells, 400 resampled groupings
each, 432,000 comparisons, reported against the ratio of the largest mean
contribution to the second largest, which a reader computes from a quantity
take-off in one line.

| P(the method changes the leader) | crosses at a ratio of | 95 pct interval | isotonic |
|---|---|---|---|
| 1 percent | **2.13** | 2.09 to 2.17 | 2.22 |
| 5 percent | **1.64** | 1.62 to 1.66 | 1.61 |
| 10 percent | **1.46** | 1.45 to 1.47 | 1.35 |

**Dispersion enters in the expected direction and is weak.** A logistic on the
log ratio alone reaches a pseudo R2 of 0.3251; adding the log dispersion of the
two materials whose order a flip would exchange takes it to 0.3288, with
coefficients -6.36 on the ratio and **+0.37** on the dispersion. The lead
needed runs from **2.05** for a tightly spread pair to **2.19** for a widely
spread one as the pair's dispersion triples. **The rule stays "about two"**,
and the dispersion term is a refinement rather than a second mechanism.

**And a real building element sits at 1.02.** Marsh, Lewis, Hattam and Allen
(in press) report that for their Concrete-Precast staircase under the ICE
recommended factors the top two products are steel bar at 42 percent and precast
concrete at 41 percent. Their per-product quantities are in supplementary
material this repository does not hold, so no bill of quantities was
reconstructed and none was invented; for the other three factor sources the
paper states only that precast concrete leads at 65 percent, which bounds the
ratio between about 1.5 and 3.1 without identifying it.

### 4.4 The building total, and the budget statement

2,500 groups against the true distributions on the same draws. W1 is the study's
own criterion applied to the OUTPUT rather than the input, which is the reason
for using it: it puts the goodness-of-fit score and the downstream error in the
same units. The Cramer distance is reported beside it as the L2 sibling --
expected CRPS against a distributional truth reduces to Cramer plus a term that
depends only on the truth, so ranking by one is ranking by the other.

| method | W1 to the true total | 95 pct interval | error at the 90th pct | believes it meets a 90 pct budget |
|---|---|---|---|---|
| Lognormal, Uniform | **0.378** | 0.364 to 0.392 | -0.356 | 91.0 pct |
| KDE, Uniform | 0.379 | 0.365 to 0.392 | -0.180 | 89.3 pct |
| Lognormal, Variable | 0.399 | 0.383 to 0.413 | -0.348 | 91.3 pct |
| KDE, Variable | 0.403 | 0.388 to 0.419 | -0.113 | 89.0 pct |
| Normal, Uniform | 0.552 | 0.538 to 0.566 | -0.088 | 86.8 pct |
| Normal, Variable | 0.580 | 0.564 to 0.598 | -0.087 | 87.5 pct |

**Every method understates the building's 90th percentile**, and they disagree
about the budget statement in both directions. The building total averages 4.0,
so an error of 0.36 at the 90th percentile is 9 percent of the building.

### 4.5 What an intervention delivers, and how likely it is to

Two strategies, both as a fraction of the WHOLE BUILDING, which is what a
designer commits to.

| method | share of iterations capped | mean saving | chance of at least 5 pct |
|---|---|---|---|
| **the truth** | -- | **0.0539** | **0.232** |
| KDE, Uniform | 0.290 | 0.0522 | 0.239 |
| KDE, Variable | 0.299 | 0.0542 | 0.245 |
| Lognormal, Uniform | 0.277 | 0.0491 | 0.229 |
| Lognormal, Variable | 0.282 | 0.0488 | 0.229 |
| Normal, Uniform | 0.366 | 0.0619 | **0.305** |
| Normal, Variable | 0.355 | 0.0601 | 0.292 |

**The share of iterations capped is the signal the old construction destroyed.**
The cap used to be each method's own 75th percentile, so exactly 25 percent of
iterations were capped under every method by construction and a method that
understates the upper tail could not say that capping buys less. It is now one
absolute value per material, the 75th percentile of the declarations a
specifier holds.

**The quantity strategy is method-independent to four decimal places.** Using
25 percent less of a material is a deterministic fraction of its own
contribution, so no distributional assumption enters; capping acts entirely
through the upper tail, which is what the methods disagree about.

### 4.6 The materials: how wrong, and in which direction

| method | error in rank-1 frequency | 95 pct interval | error in contribution | bias in contribution | names the true leader |
|---|---|---|---|---|---|
| Lognormal, Uniform | **0.0800** | 0.0781 to 0.0818 | 0.1266 | **-0.038** | 36.2 pct |
| KDE, Uniform | 0.0816 | 0.0798 to 0.0834 | 0.1314 | -0.014 | 37.9 pct |
| KDE, Variable | 0.0825 | 0.0799 to 0.0849 | 0.1262 | **+0.003** | **53.2 pct** |
| Lognormal, Variable | 0.0852 | 0.0827 to 0.0877 | 0.1201 | -0.027 | 50.1 pct |
| Normal, Uniform | 0.1151 | 0.1132 to 0.1172 | 0.1640 | **+0.044** | 22.9 pct |
| Normal, Variable | 0.1193 | 0.1170 to 0.1215 | 0.1683 | **+0.050** | 37.0 pct |

**The four non-normal methods span six percent of each other and the normal is
forty percent worse than any of them.** On a win share over 10,000 materials --
how often a method is CLOSEST to the truth, which is a count and so does not
inherit every material's noise -- the best is `KDE, Variable` at 0.215 (0.207 to
0.226) against the one-in-six of 0.167 six methods would give by chance.

**The spread matters more than the average.** The error on any ONE material has
a standard deviation of **0.234 to 0.301** and a 99th percentile of **0.895 to
1.259**: for one material in a hundred the method is wrong by more than that
material's entire expected contribution.

**On the upper tail they fail together.** Every method understates a material's
95th percentile, by 0.086 under `KDE, Uniform` and 0.219 under
`Normal, Variable`.

**And they fail on the same materials.** Per-material errors correlate 0.892 to
0.970 between methods that share a weighting scheme and only 0.581 to 0.714
across weighting schemes; all six err in the same direction on 51.2 percent of
materials. The dominant axis of disagreement is the WEIGHTS, the three families
make nearly the same error on the same material, and choosing a different family
does not hedge the risk.

### 4.7 Knowing market shares against guessing them

**This is not a comparison of two weighting schemes and must not be read as
one.** A synthetic dataset's weights give each mode its true market share --
signal -- and then split that share among the points inside the mode by a flat
Dirichlet, which is noise the real world does not have. The oracle keeps the
first and replaces the second with an equal split. It is not available to a
practitioner and is not proposed as a method.

1,200 groups, mean absolute error in a material's estimated contribution:

| family | uniform | this study's drawn weights | knowing the shares |
|---|---|---|---|
| KDE | 0.1285 | 0.1215 | **0.1064** |
| Lognormal | 0.1243 | 0.1168 | **0.1007** |
| Normal | 0.1620 | 0.1629 | 0.1550 |

On the rank-1 frequency the last column gains as much again -- 0.0815, 0.0818,
**0.0717** for the kernel estimate -- **and the middle column gains nothing at
all**. So variable weighting is better than uniform on the magnitude, a wash on
the ranking, and would be better than both if the shares were known. **The gap
between the last two columns is the cost of the stand-in**, which is a
limitation of this generator rather than a property of weighting, and it is the
strongest argument the study has for real production volumes.

### 4.8 What common random numbers were worth

Three quantities per output, over 300 pLCAs and all fifteen method pairs. Each
figure is the change for the most-affected of the four materials.

| output | two methods, shared draws | two methods, own draws | ONE method, twice |
|---|---|---|---|
| estimated contribution | 0.1819 | 0.1828 | 0.0095 |
| 95th percentile of it | 0.4152 | 0.4153 | 0.0239 |
| chance of being largest | 0.1275 | 0.1274 | 0.0073 |
| contribution to variance | 0.0898 | 0.0914 | 0.0132 |
| **top contributor changes** | **0.548** | **0.559** | **0.0367** |

**The middle column is what the study measured before and it was not
inflated**: every ratio of it to the first is between 0.99 and 1.02. Where
pairing matters is the last row, an argmax, where 3.67 percent of comparisons
name a different top contributor with no model difference at all.

### 4.9 The group size, and an interval on every headline

**More materials does not dilute the effect.** From two materials to twelve, at
equal intensities: a material's own estimated contribution moves 0.069 to 0.095
when the method changes, its share of the building total moves 0.0199 to 0.0077,
and the probability that two methods name a different largest contributor RISES
from 0.448 to 0.656. Only the share-of-total outputs dilute, so **the paper's
sentence has to name the output**.

**The flip thresholds the previous stage published rise with the group size
under the maximum and are flat under the mean**, which is an artifact of the
summary: a maximum over twelve materials is drawn from more chances than a
maximum over two. At four materials the recomputed crossings are 0.0012, 0.0087
and 0.0216 against the published 0.0018, 0.011 and 0.025, the same two
significant figures given that these use 300 resampled groupings against 2,500.
**The published constants stand and the weighting-risk probabilities that read
them are not recomputed.**

**Every NRMSE now carries a cluster bootstrap over pLCAs.** A material's rank-1
frequency is **1.042** (1.034 to 1.051): above 1 means the root mean squared
difference between two methods exceeds the standard deviation of that output
across every material and method. The uncertainty index is the lowest of the
main outputs at **0.503** (0.492 to 0.515).

## 5. Numbers that moved

**Two changes moved published numbers and both are in one table.**

**Common random numbers** change which variates each method sees, so every row
of the 60,000-row results table moves by Monte Carlo noise and no aggregate
moves. A material's rank-1 frequency changes by a mean of **0.0047** per row and
at most 0.0288, which is 4.4 percent of that column's standard deviation; the
largest relative move in any column's MEAN is 0.34 percent; a material's
estimated contribution goes from a mean of 1.042702 to 1.042689.

**The absolute specification cap** moves every `capecc_` column, and this one is
a real change rather than noise. The mean reduction from capping goes from
**-0.8623 to -0.8349** and the mean percentage reduction from -0.1712 to
-0.1698. The rank-frequency columns move much further, because the
`(1 - capecc)` divisor had to go with the cap it normalized: it scaled a count
by 1 / 0.25, which was exact only because the old cap bound in exactly 25
percent of iterations for every material. `capecc_rank_1` therefore goes from a
mean of 0.684 to **0.193**, which is a change of definition and not of
substance -- the old column was a frequency over capped iterations rescaled by a
constant that no longer applies, and the new one is the share of all iterations
in which capping this material both bound and gave the largest reduction.

**Seven figures drawn from that table are redrawn, and two are rebuilt.**

**Nothing else moved.** The previous stage's flip calibration tables are BYTE
IDENTICAL, because their random streams are spawned from the seed sequence
rather than taken from the consumed stream, and the eight regression fixtures
pass unchanged. The aggregate truth-run figures are stable to four decimal
places across runs with different draws, which is itself the check that 2,500
groups of 10,000 iterations is enough.

## 6. What is still open

### Carried forward, still open, owned elsewhere

| Item | Owner |
|---|---|
| Shapiro-Wilk versus Shapiro-Francia, and the reduction of the characteristic set to three to five survivors | 2f |
| The metric set: **this stage recommends reporting the five statements above, with the uncertainty index added, since it is the steadiest output measured and appears in no table, figure or section**. The `(1-capecc)` divisor is partly resolved here and should still be reviewed | 2g |
| The profile-likelihood guard sweep; the Dirichlet concentration sweep, which should vary the BLOCK STRUCTURE and not only the parameter; multiple weight realizations; the deduplicated variant; the mode-share coupling | 2h |
| Every figure brought to the style guide, **including the Unicode minus, which this stage fixed for the figures that call the style module and which the older ones still carry**; the figure manifest; one results table is 92 MB | 3 |
| A real-building anchor, if citing Marsh et al. (in press) is not enough. **This stage used that paper's stated contribution percentages and found one exactly stated pair at a ratio of 1.02** | 2i, optional |
| An industry-average EPD as a direct estimate of the market-weighted mean | unowned |

### Resolved here

Common random numbers and dependent sampling; the materials-per-pLCA sweep;
resampled groupings; bootstrap intervals on every headline percentage and
NRMSE; the probabilistic LCA against the true distributions; the smoke-run
guard, which was Stage 3's item; and the specification cap, which no stage had
identified as broken.

### New, and worth writing down

**The design comparison is a new capability, not just a new result.** The study
could not compare two designs at all, which is what all three of its
comparative-LCA citations are about. The machinery is now there, with dependent
sampling, and it supports more than the one-material swap: more than two
options, and rank acceptability, would both fall out of it.

**The truth run cannot arbitrate uniform against variable weighting**, and the
oracle experiment in section 4.7 is the closest this study can come. Any
statement comparing the two weighting schemes at the decision level has to be
framed as knowing shares against guessing them.

**The repository's history is 2.5 GB**, against the 391 MB recorded when its
size was accepted. The growth is large tables re-stored whole on every commit,
this stage added about 33 MB of them, and the rewrite that would shrink it was
declined because it would break the published deposit. **The next Claude Code
session should not act on this**; it is here so the number is somewhere.

## 7. Inputs and outputs

**Read.** The synthetic corpus, the parents recovered by replaying the
generator, and the mode labels recovered the same way; the frozen raw empirical
extract, for the one real dataset a figure illustrates and for the size mix the
post-stratified numbers reweight by; and the published staircase paper, for the
one real contribution ratio available.

**Written.** One source module and its test file; nineteen new cells and two
rebuilt figures in the third notebook; twenty-three new result tables; nineteen
decisions in the project's decision log, numbered 105 through 123; manuscript
discrepancy entries 96 through 113, with entry 87 marked resolved; and this
file.

**Not touched.** The generator, the synthetic corpus, the empirical extract, the
fitting methods, the scoring criterion, the published flip thresholds, and the
manuscript.

## 8. Next stage

**Stage 2f**, the metric reduction: resolve Shapiro-Wilk versus Shapiro-Francia,
then cut the characteristic set to three to five survivors.

**What this stage hands it.** A characteristic that predicts the distance
between a fitted curve and its target but not the error in the ANSWER is not
worth keeping, and the per-material errors against the truth are on disk for
exactly that test. Dataset size also remains the only mechanism anything has
found, so a reduction that ends with size and little else would be consistent
with everything measured rather than a failure of the reduction.

**Stage 2g owns the metric set, and the recommendation from here is specific.**
Report the five statements; lead with the comparison, because it is the decision
and the answer is a null; add the uncertainty index, which is the steadiest
output measured and is reported nowhere; and state the specification result,
because it is where the choice of method costs the most -- a normal overstates
the chance of achieving a 5 percent building-level saving by seven points.
