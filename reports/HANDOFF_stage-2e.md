# HANDOFF stage-2e - The pLCA construction

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

**These ten sentences are what Stage 2e contributes to the manuscript.** Each
is a claim the paper can make, with the number that supports it. Everything
below this page is the working record.

**The one that should lead the results**

1. **At the level of the ANSWER, the choice between a kernel estimate and a
   three-parameter lognormal does not matter, and the choice of a normal does.**
   Running every probabilistic LCA twice on the same random draws -- once with
   the fitted models and once with the distributions the data was actually drawn
   from -- the error in a material's chance of being the largest contributor is
   **0.0800** for the lognormal under uniform weights, **0.0816** and **0.0825**
   for the two kernel estimates, **0.0852** for the lognormal under variable
   weights, and **0.1151** and **0.1193** for the two normals. The four
   non-normal methods span 6 percent of each other; the normal is **40 percent
   worse** than any of them.
2. **No method recovers the answer.** The best names the material that truly
   contributes most **53 percent** of the time, against 25 percent for a coin
   toss among four, and misstates a material's estimated contribution by
   **0.12** where every material contributes 1.00.

**On what the ranking result depends on**

3. **A material must lead the next by about a factor of 2 before the choice of
   UQ method cannot change which one leads.** The probability of a changed
   leader crosses 1 percent at a top-two contribution ratio of **2.13**
   (95 percent interval 2.09 to 2.17), 5 percent at **1.64** and 10 percent at
   **1.46**. The ratio is the largest material's mean contribution divided by
   the next largest, which a reader computes from a quantity take-off in one
   line.
4. **And a real building element sits at 1.02.** Marsh, Lewis, Hattam and Allen
   (in press) report that for their Concrete-Precast staircase under the ICE
   recommended factors the top two products are steel bar at 42 percent and
   precast concrete at 41 percent. A real design can sit within a percentage
   point of a tie, which is exactly where the choice of UQ method decides the
   ranking.
5. **Concentration protects the ranking and does nothing for the numbers.** At
   four materials the median change in a material's estimated contribution is
   **0.188** at equal intensities and **0.166** at 100:1, an 11 percent decline
   while the probability that the leader changes falls from 0.546 to zero.
   Pooled over group sizes the same two figures are 0.212 and 0.210, no decline
   at all.
6. **More materials does not dilute the effect.** From two materials per
   probabilistic LCA to twelve, a material's own estimated contribution moves
   **0.069 to 0.095** when the UQ method changes, its share of the building
   total moves **0.0199 to 0.0077**, and the probability that two methods name a
   different largest contributor **rises from 0.448 to 0.656**. Only the
   share-of-total outputs dilute.

**On the machinery, which the paper states rather than argues**

7. **The comparison between UQ methods is now paired**, one uniform variate per
   material per iteration through every method's inverse CDF, as Henriksson et
   al. (2015) and Heijungs (2021) recommend and Marsh et al. (in press) do.
   **It was a refinement and not a repair**: the sampling noise is 4 to 15
   percent of the difference between two methods and adds nearly orthogonally,
   so no previously measured median moved by more than 2 percent. Where it
   matters is the argmax: **3.67 percent** of comparisons named a different top
   contributor with no model difference at all, and that floor is now zero.
8. **Every headline percentage and every NRMSE now carries a bootstrap
   interval**, resampling probabilistic LCAs rather than rows. The headline
   output's NRMSE is **1.042** (1.033 to 1.051): the choice of UQ method moves a
   material's rank-1 frequency by more than the standard deviation of that
   quantity across every material and method.
9. **Concentration fixes the ranking and leaves the error in the numbers
   exactly where it was, and two independent measurements say so.** Running the
   same probabilistic LCAs against the true distributions with the leading
   material at ten times every other, all six methods name the true largest
   contributor in **every** group and the error in a material's rank-1 frequency
   falls from about 0.09 to about 0.008 -- while the error in its estimated
   contribution stays at 0.12 to 0.17, exactly where it was at equal
   intensities. A practitioner with one dominant material can trust the ranking
   and still cannot trust the magnitude.
10. **The steadiest output is still the one that sets data-collection
   priorities.** The uncertainty index has the lowest NRMSE of the main outputs
   at **0.503** (0.491 to 0.515), and the study computes it and reports it
   nowhere.

**What must travel with the ranking numbers, in the same paragraph and not a
footnote.** Every material in this study carries a use intensity of 1.0, so the
contributions are exchangeable and a ranking is as fragile as it can be made.
Sentences 1, 2 and 5's contribution figures do not depend on that; the flip
probabilities are an upper bound because of it. And material use intensity is
deterministic here, while a real quantity take-off carries its own uncertainty,
which in practice can exceed the coefficient uncertainty this paper is about.

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

Six things, in a stated order because one is a prerequisite for a published
number.

**First, the sweep over materials per probabilistic LCA**, because the flip
thresholds the previous stage published -- the probability that the
top-contributing material changes crossing 1 percent at a relative Wasserstein
distance of 0.0018, 5 percent at 0.011 and 10 percent at 0.025 -- are all
conditional on four materials of equal material use intensity.

**Then the material use intensity sweep**, added by the author mid-stage as the
more consequential of the two axes, to be sampled on the simplex rather than in
absolute units, reported against an observable rather than against a
concentration parameter, anchored to a real design if one was available, and
crossed with the group-size sweep rather than run beside it.

**Then**: a check that makes it impossible for a smoke run to overwrite the
project's most expensive table; common random numbers across UQ methods, reusing
the tested implementation rather than writing another; resampled groupings with
bootstrap intervals on every headline percentage and NRMSE; and the experiment
the previous two stages both identified as the most valuable one left, the
probabilistic LCA run against the TRUE distributions the synthetic data was
drawn from.

---

## 3. What was done

**One new source module, `src/plca.py`, with 36 tests.** It holds the
common-random-numbers draw, the ten pLCA outputs in the notebook's own
definitions, material use intensity as a share vector, the resampled groupings,
the cluster bootstrap, NRMSE with an interval, and the run against the true
parents. Everything the notebook does below is one call into it.

**Notebook 3 gained five sections -- eleven cells -- and two figures**, and its
main pLCA loop now uses common random numbers. Nothing else in the notebook changed.

**The whole test suite is 423 tests and all pass**, including the eight
regression fixtures that pin the empirical metrics, the synthetic metrics and
all six goodness-of-fit scores. That is the check that the fitting, the corpus
and the empirical arm were not touched.

---

## 4. What the measurements say

### 4.1 Common random numbers were worth about one percent

Three quantities per output, over 300 probabilistic LCAs and all fifteen pairs
of the six methods. Each figure is the change for the most-affected of the four
materials, which is the one a practitioner is deciding about, and every material
contributes a mean of 1.00, so they read directly.

| output | two methods, shared draws | two methods, own draws | ONE method, twice |
|---|---|---|---|
| estimated contribution | 0.1819 | 0.1828 | 0.0095 |
| 95th percentile of it | 0.4152 | 0.4153 | 0.0239 |
| chance of being largest | 0.1275 | 0.1274 | 0.0073 |
| contribution to variance | 0.0898 | 0.0914 | 0.0132 |
| **top contributor changes** | **0.548** | **0.559** | **0.0367** |

**The middle column is what the study measured before and it was not inflated**:
every ratio of it to the first column is between 0.99 and 1.02. The sampling
noise is 4 to 15 percent of the model difference and adds nearly orthogonally,
which is why it never moved a median.

**The last row is where it does matter.** With no model difference at all, 3.67
percent of comparisons name a different largest contributor, because four
near-exchangeable materials are often statistically tied at the top and a
tie-break is a coin toss. That is a floor under any flip statistic read off the
old table, and common random numbers remove it exactly.

**This confirms from the other side the withdrawal the previous stage made.** It
had reported that running one method twice changes "the answer" 5.33 percent of
the time and built a case on it; the author rejected that framing repeatedly and
was right, because the statistic counts a label landing elsewhere between two
statistically tied materials rather than any reported quantity moving.

### 4.2 More materials does not dilute the effect

Equal intensities, 400 resampled groupings per size, fifteen method pairs each.

| materials per pLCA | 2 | 3 | 4 | 6 | 8 | 12 |
|---|---|---|---|---|---|---|
| P(top contributor changes) | 0.448 | 0.544 | 0.546 | 0.624 | 0.634 | 0.656 |
| change in estimated contribution | 0.069 | 0.076 | 0.081 | 0.089 | 0.089 | 0.095 |
| change in share of the total | 0.0199 | 0.0204 | 0.0179 | 0.0137 | 0.0108 | 0.0077 |
| change in chance of being largest | 0.0684 | 0.0790 | 0.0727 | 0.0585 | 0.0476 | 0.0362 |

The intervals are cluster bootstraps over probabilistic LCAs and are about 0.02
wide on the flip rates.

**The expectation stated in the stage prompt was that the effect would dilute,
because each material's share of the total shrinks. Only the share-of-total
outputs do.** A material's own estimated contribution does not dilute at all and
grows slightly, because a bigger group has more chances to contain a dataset the
methods disagree about; and the probability that two methods name a different
largest contributor rises, because more materials means more chances of a
near-tie at the top.

**So the paper's sentence has to name the output.** Written without naming it,
the natural sentence is wrong in the direction that flatters the study.

### 4.3 The ratio at which a leading material makes the ranking safe

Intensities drawn on the simplex from a symmetric Dirichlet, concentration from
200 down to 0.15, plus deterministic checkpoints at 1:1, 2:1, 10:1 and 100:1,
crossed with the six group sizes: 72 cells, 400 groupings each, 432,000
comparisons.

**Reported against the ratio of the largest mean contribution to the second
largest**, never against the Dirichlet concentration, which means nothing to a
reader and whose implied dominance changes with the number of materials, so
reporting against it would confound the two sweeps.

| P(the UQ method changes the leader) | crosses at a ratio of | 95 pct interval | isotonic fit |
|---|---|---|---|
| 1 percent | **2.13** | 2.09 to 2.17 | 2.22 |
| 5 percent | **1.64** | 1.62 to 1.66 | 1.61 |
| 10 percent | **1.46** | 1.45 to 1.47 | 1.35 |

The isotonic fit assumes only that the probability does not rise as the leader
pulls away, so its agreement says the crossings are a property of the data
rather than of the link function. **The crossing moves little with the group
size**: the 1 percent crossing is 1.90 at two materials and 2.34 at twelve.

The observed rate in equal-count bins runs 0.576 at a ratio of 1.00, 0.410 at
1.10, 0.290 at 1.17, 0.150 at 1.31, 0.049 at 1.59, 0.042 at 2.00, 0.007 at 2.45
and 0.000 from 10 upward.

**AND CONCENTRATION DOES NOTHING FOR THE NUMBERS.** At the study's own four
materials the median change in a material's estimated contribution is **0.188**
at equal intensities, 0.186 at 2:1, 0.175 at 10:1 and **0.166** at 100:1: an 11
percent decline in the magnitude while the ranking goes from a coin toss to
certainty. Pooled over the six group sizes the same four figures are 0.212,
0.213, 0.193 and 0.210, which is no decline at all. **That contrast is the
figure and it is the finding.**

The comparison is in absolute units and that is legitimate here, because the
intensity vector is normalized to a mean of 1.0 in every cell, so the building's
total mean contribution is the number of materials whatever the concentration.
Section 4.6a says what changes if the denominator is the leading material's own
contribution instead.

### 4.4 The anchor, and what could not be anchored

Marsh, Lewis, Hattam and Allen (in press), "Uncertainty characterisation for
construction products and comparative metrics for probabilistic building LCA",
compares four staircase designs whose quantities come from project BIM models.
**Its per-product quantities are in supplementary material this repository does
not hold**, so no bill of quantities was reconstructed and none was invented.

What the paper states in its own text is the contribution of each product to the
total, which is the ratio this sweep is reported against. For the
**Concrete-Precast** design under the **ICE recommended** factors the two top
contributors are **steel bar at 42 percent and precast concrete at 41 percent**,
a top-two ratio of **1.02**. Under the other three factor sources the paper
states only that precast concrete leads at 65 percent while steel bar runs
between 21 and 42 percent across all four sources, which bounds the ratio
between about 1.5 and 3.1 without identifying it.

**One exactly stated pair is enough for the point the paper needs**: the
equal-intensity construction this study uses is not an extreme, because a real
building element can put two products within a percentage point of each other.

### 4.5 The flip thresholds by group size, and why they rise

The previous stage's thresholds were calibrated on groups of four and the stage
prompt required them to be recomputed at every size. The flip RATE does rise
with the group size, 0.106 at two materials to 0.188 at twelve. The thresholds
do not fall as predicted, and the reason is an aggregation artifact worth
stating in the text.

The model distance is a property of a MATERIAL and the decision is a property of
the GROUP, so the per-material distances have to be summarized. The published
constants use the MAXIMUM, and a maximum over twelve materials is drawn from
more chances than a maximum over two.

| materials | 2 | 3 | 4 | 6 | 8 | 12 |
|---|---|---|---|---|---|---|
| maximum, 1 pct crossing | 0.0015 | 0.0020 | 0.0012 | 0.0021 | 0.0022 | 0.0032 |
| maximum, 5 pct crossing | 0.0098 | 0.0108 | 0.0087 | 0.0126 | 0.0124 | 0.0157 |
| mean, 1 pct crossing | 0.0011 | 0.0012 | 0.0006 | 0.0010 | 0.0010 | 0.0011 |
| mean, 5 pct crossing | 0.0065 | 0.0060 | 0.0041 | 0.0052 | 0.0047 | 0.0049 |

**Under the mean the thresholds are constant in the group size.** Under the
maximum they roughly double from two materials to twelve, and that is the
summary drifting rather than the probabilistic LCA changing.

At four materials the recomputed crossings are 0.0012, 0.0087 and 0.0216 against
the published 0.0018, 0.011 and 0.025. These use 300 resampled groupings against
the published values' 2,500, and the published values already carry a stated 30
percent interval width. **So the published constants stand, the per-dataset
weighting-risk probabilities that read them are NOT recomputed, and the paper
states the conditionality with the measured dependence beside it.**

### 4.6 The pLCA against the truth

Every group run twice on the same uniform variates, once with the fitted models
and once with the parents the datasets were drawn from, so the difference is the
error the fitted model causes with no Monte Carlo noise in it at all. 2,500
groups, 10,000 draws each, cluster-bootstrap intervals over groups.

**The truth is the market-weighted parent**, because a probabilistic LCA of what
gets built is a statement about the population of products weighted by how much
of each is produced, and it is the one population all six methods can be scored
against on equal terms.

| method | error in rank-1 frequency | 95 pct interval | error in estimated contribution | names the true leader |
|---|---|---|---|---|
| Lognormal, Uniform | **0.0800** | 0.0781 to 0.0818 | 0.1266 | 36.2 pct |
| KDE, Uniform | 0.0816 | 0.0798 to 0.0834 | 0.1314 | 37.9 pct |
| KDE, Variable | 0.0825 | 0.0799 to 0.0849 | 0.1262 | **53.2 pct** |
| Lognormal, Variable | 0.0852 | 0.0827 to 0.0877 | 0.1201 | 50.1 pct |
| Normal, Uniform | 0.1151 | 0.1132 to 0.1172 | 0.1640 | 22.9 pct |
| Normal, Variable | 0.1193 | 0.1170 to 0.1215 | 0.1683 | 37.0 pct |

**The four non-normal methods span 0.0800 to 0.0852, six percent of each other,
and the normal is forty percent worse than any of them.** On a win share over
10,000 materials -- how often a method is CLOSEST to the truth, which is a count
and so does not inherit every material's noise -- the best is `KDE, Variable` at
0.215 (0.207 to 0.226) against the one-in-six of 0.167 that six methods would
give by chance. Reweighting to the size mix of the real EC3 categories moves
every figure by less than 0.003.

**Nobody recovers the answer.** The best method names the material the truth
says is the largest contributor 53 percent of the time, against 25 percent for a
coin toss among four, and the best error in a material's estimated contribution
is 0.12 where every material contributes 1.00.

**The uniform-weighted methods look much better against the SAMPLING parent** --
`KDE, Uniform` scores 0.0609 rather than 0.0816 -- **and the variable-weighted
ones worse**, `KDE, Variable` 0.1012 rather than 0.0825. That gap is
definitional and not an error of estimation: it is the difference between the
population a method estimates and the population a building is about. Both are
reported.

### 4.6a The truth run and the dominance sweep agree, by two routes

Section 4.3 found that concentration kills the flip probability and leaves the
magnitude alone by comparing the methods with EACH OTHER. The truth run says the
same thing by comparing each method with the RIGHT ANSWER, which is a different
measurement and could have disagreed. Same pLCA against the true parents at
three intensity settings, 600 groups each, with the leading material at 1, 2 and
10 times every other:

| | 1:1 | 2:1 | 10:1 |
|---|---|---|---|
| names the TRUE largest contributor | 0.23 to 0.51 | 0.95 to 0.98 | **1.00, all six methods** |
| error in a material's rank-1 frequency | 0.078 to 0.119 | 0.056 to 0.076 | **0.0075 to 0.0091** |
| error in its estimated contribution | 0.115 to 0.161 | 0.116 to 0.163 | **0.119 to 0.173** |

**At ten to one every method names the true leader in every group and the error
in the rank-1 frequency falls tenfold, while the error in the estimated
contribution does not move at all.**

**Why the absolute comparison is the right one, and it is a property of the
construction rather than an assumption.** The intensity vector is normalized to
a mean of 1.0 in every cell, so the building's total mean contribution is the
same number -- the number of materials -- whatever the concentration. An
absolute error of 0.12 is therefore the same share of the building at 1:1 as at
10:1. Read instead as a fraction of the LEADING material's own contribution the
same error does fall, because that material is larger, so the paper must say
which denominator it is using.

**What the pair of results licenses.** A practitioner whose design has one
dominant material can trust the ranking under any of these methods and still
cannot trust the magnitude, which is what a carbon budget is written in.

### 4.7 An interval on every headline

The study reported an NRMSE between the six methods for every pLCA output and
attached uncertainty to none of them. Every one now carries a cluster bootstrap
over probabilistic LCAs, because the four materials of a group share its total
and its variates and a bootstrap over rows comes back more than twice too
narrow.

The headline output, a material's rank-1 frequency, has an NRMSE of **1.042**
(1.033 to 1.051). **Above 1 means the root mean squared difference between two
UQ methods exceeds the standard deviation of that output across every material
and method** -- the choice of method moves the answer by more than the spread it
is trying to describe. The lowest of the main outputs is the uncertainty index
at 0.503 (0.491 to 0.515), which the study computes and reports nowhere.

---

## 5. Numbers that moved

**One table moved and every move in it is Monte Carlo noise.** Installing common
random numbers changes which variates each method sees, so every row of the
60,000-row pLCA results table changes.

| | |
|---|---|
| Per row, a material's rank-1 frequency | mean change **0.0047**, 99th percentile 0.0157, largest 0.0288, which is 4.4 percent of that column's standard deviation |
| The largest relative move in ANY column's MEAN | **0.34 percent**, on the mean frequency with which a capped material ranks fourth, whose mean is 0.0039 |
| A material's estimated contribution | mean 1.042702 to 1.042689 |
| The integer rank columns | mean change of 0.25 of a rank, which is the tie-break above and not a change in any quantity |

**Seven figures drawn from that table are redrawn.** Nothing else moved: the
previous stage's flip calibration tables are BYTE IDENTICAL, because their random
streams are spawned from the seed sequence rather than taken from the consumed
stream, and the eight regression fixtures pass unchanged.

**The notebook was run end to end twice, and the second run reproduced the
first's tables byte for byte in content** -- every one of them, including the
60,000-row results table and the 432,000-row sweep -- with only file timestamps
differing. The second run existed to improve a figure and to add one table, and
what it settles beyond that is that the whole 39-minute analysis is reproducible
from its seed.

**One table changed in its last decimal digit**, the post-stratified flip rate,
because the empirical size shares it reweights by are now measured from the arm
rather than written as a constant. 0.11474905550369241 becomes
0.1147490555036924.

---

## 6. What is still open

### Carried forward, still open, owned elsewhere

| Item | Owner |
|---|---|
| Shapiro-Wilk versus Shapiro-Francia, and the reduction of the characteristic set to three to five survivors | 2f |
| The `(1-capecc)` divisor; magnitude-based companion metrics; the sensitivity of the headline rank-1 frequency; **and reporting the uncertainty index at all, which this stage found to be the steadiest output the analysis produces and which appears in no table, figure or section** | 2g |
| The profile-likelihood guard sweep reporting the fitted-model spread ratio at every value; the Dirichlet concentration sweep, which should vary the BLOCK STRUCTURE and not only the concentration; multiple weight realizations; the deduplicated variant; the mode-share coupling | 2h |
| Every figure brought to the figure style guide; the figure manifest; **one results table is 96 MB** | 3 |
| The real-building anchor, if the author decides citing Marsh et al. (in press) is not enough. **This stage used that paper's stated contribution percentages and found one exactly stated pair at a ratio of 1.02; its per-product quantities are in supplementary material this repository does not hold** | 2i, optional |
| An industry-average EPD as a direct estimate of the market-weighted mean, which the author's companion paper already uses under another name | unowned |

### Resolved here

Common random numbers in the study's own pLCA; dependent sampling, which is the
same change; the materials-per-pLCA sweep; resampled groupings; bootstrap
intervals on every headline percentage and NRMSE; the probabilistic LCA against
the true parents; and the smoke-run guard, which was Stage 3's item and was done
here because this stage reruns the artifact it protects.

### New, and small

**The sweep's own table is 432,000 rows and is written as Parquet**, which is
33 MB against 39 MB as gzipped CSV. That is a 16 percent saving and not the
large one it might sound like, because float64 does not compress well either
way; it is written that way for the typing and the read speed as much as the
size, and Parquet is what this project already chose for its large tidy tables.
**The 96 MB table noted above is a different one and is still Stage 3's.**

**The equal-intensity construction is now measured rather than assumed to be
conservative.** It is the most fragile case for a ranking and it makes no
difference to the magnitude outputs, which is the strongest available defense of
the study's own design and belongs in the text as one.

---

## 7. Inputs and outputs

**Read.** The synthetic corpus and the parents recovered by replaying the
generator; the frozen raw empirical extract, for the one real dataset a figure
illustrates and for the size mix the post-stratified numbers reweight by; and
the published staircase paper, for the one real contribution ratio available.

**Written.** One source module and its test file; eleven new cells and two new
figures in notebook 3; sixteen new result tables; nine decisions in the project
brief's decision log, numbered 105 through 113; manuscript discrepancy entries 96
through 104, and entry 87 marked resolved; and this file.

**Not touched.** The generator, the synthetic corpus, the empirical extract, the
fitting methods, the scoring criterion, the published flip thresholds, and the
manuscript.

---

## 8. Next stage

**Stage 2f**, the metric reduction: resolve Shapiro-Wilk versus Shapiro-Francia,
then cut the characteristic set to three to five survivors with a multivariate
model of the goodness-of-fit score and of which method wins.

**Two things this stage hands it.** The first is that the decision-level result
is now available as a target: a characteristic that predicts the fitted CDF's
distance but not the error in the answer is not worth keeping, and the table of
per-material errors against the truth is on disk for exactly that. The second is
that dataset size remains the only mechanism anything has found, so a reduction
that ends with size and little else would be consistent with everything measured
so far rather than a failure of the reduction.

**Stage 2g** should take the uncertainty index seriously. It is the steadiest
output the analysis produces, it is the one a practitioner acts on when deciding
where to spend effort collecting better data, and it is computed and reported
nowhere.
