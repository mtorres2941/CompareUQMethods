# Stage 2j: let the UQ method vary by material

**Corpus** `corpus_2026-09-25`. **Every number here is the SYNTHETIC arm**, where
each point's weight is the TRUE market share of its product group, exact to
1.1e-16 -- not a guess (decision 212). 2,500 pLCA groups run twice on the same
uniform variates, once with fitted models and once with the true parents, so the
difference carries no Monte Carlo noise. 57 policy columns. Both facts are
stamped on every table this stage writes.

**Rebuilt 2026-09-30 from the current tables**, not patched. An earlier version
carried three claims its own tables contradicted; those are corrected below and
recorded in decisions 222 to 224.

---

## What this found

**1. Switching family by size beats any fixed method a reader can actually
use, by about one percent.** Uniform weights throughout, kernel estimate at or
above the cutoff and three-parameter lognormal below: pooled error **0.2324**
against the best uniform-weighted fixed method's 0.2392, or **+2.9 percent**.
Claim by claim against the best fixed method *for that claim*, it wins **9 of
15** with a median of **+1.4 percent**.

**2. Not knowing market shares costs about three points out of twenty-three.**
Give the same rule the true shares above the cutoff and the pooled error falls
from **23.24 to 20.25 percent** of the true level: **2.98 points**, or 12.8
percent of what was there. The absolute form is the honest one -- a
probabilistic LCA is wrong by about 23 percent either way. Weighting is not the
big factor, which is a reassuring result rather than a disappointing one.

**3. The cutoff barely matters: publish 40 to 170 declarations.** Swept every 10
from 10 to 200, plus both degenerate ends. Everything from 40 to 170 is
statistically indistinguishable from the best, in one unbroken run, and the
whole sweep from 3 to 10,000 is worth 0.73 points against the 2.98 that market
shares are worth.

**4. Market share is not the problem; too few declarations to use it is.**
Knowing the true shares makes a fit WORSE below about 80 declarations and much
better above 100. With two declarations carrying ninety percent of the weight
you over-index on two values that may not represent their own product group.

**TWO BANDS, TWO QUESTIONS, AND THEY ARE DIFFERENT NUMBERS.** Where to switch
FAMILY -- kernel estimate against three-parameter lognormal -- is anywhere from
**40 to 170** declarations; the cost is flat across all of it. Where knowing
MARKET SHARE stops hurting and starts helping is a much tighter **80 to 100**:
significantly harmful below 80, significantly helpful above 100, statistically
zero in between. A reader who absorbs the first and carries it into the second
will be wrong, so the paper must keep them apart.

**Needs an author decision: nothing.**

---

## 1. Two rules, and only one of them is a method

                                             pooled error over 16 claims
    FEASIBLE: uniform weights, family switches at the cutoff      0.2324
      KDE, uniform weights                                        0.2392
      Lognormal, uniform weights                                  0.2396
      Normal, uniform weights                                     0.3144

    KNOWN SHARES: the same switch, true market shares above it    0.2025
      KDE, market weights                                         0.2293
      Lognormal, market weights                                   0.2365

**Only the first is implementable.** Nobody publishes market shares, so uniform
weighting is not a choice a practitioner makes -- it is the only option -- and
what they CAN choose is which family to fit (decision 216). The second rule is a
value of information, and the gap between the two is what that information is
worth.

**So what.** A reader with a set of EPDs and nothing else can count them and
pick a curve, and that is worth about one percent of the error in a
probabilistic LCA. Obtaining real production volumes would be worth four times
as much.

    Reproduce: outputs/tables/TABLE_MixedPolicyRanking.csv

## 2. Where the cutoff goes, and why there are two numbers

Swept every 10 from 10 to 200, plus 3, 300, 500, 1,000, 3,000 and 10,000, at
6,000 bootstrap resamples over pLCA groups.

    cutoff     3    10    20    30    40    50    70   100   130   170   200   1000  10000
    pooled  .2392 .2364 .2337 .2331 .2326 .2324 .2325 .2324 .2324 .2326 .2328 .2348  .2396

**THE RANGE TO PUBLISH IS 40 TO 170.** Those are the cutoffs whose cost cannot
be told apart from the best, as one unbroken run: 30 falls out below, 180 falls
out above, and nothing in between does. It is the direct answer to the question
a practitioner asks -- which cutoffs can I use without paying a measurable
penalty.

**A second number is printed beside it and is NOT the one to publish.** The
bootstrap interval of the curve's own argmin is [50, 140]. That answers a
different question -- where does the single lowest point land on resampling --
and a practitioner is not choosing the optimum, they are choosing something
good enough. It is narrower because the curve genuinely tilts at the edges of
the band (40 sits 0.00025 above the minimum and 170 sits 0.00017 above), not
because of any artifact; that was checked against a constructed flat region,
where the argmin interval recovers the region exactly (decision 224).

**And the band is conservative.** Its reference is the argmin of the same data,
so differences measured against it are biased slightly positive, which makes the
band if anything too narrow.

**The sweep is self-checking.** The corpus holds 3 to 9,999 declarations, so a
cutoff of 3 assigns every dataset the kernel estimate and a cutoff of 10,000
assigns every dataset the lognormal. Both reproduce those fixed methods to
**0.00e+00**, printed on every run.

**So what.** Anywhere from forty to a hundred and seventy declarations works.
Getting the number exactly right is worth about a quarter of what having the
rule at all is worth.

    Reproduce: outputs/tables/TABLE_MixedPolicyThreshold.csv, family=feasible

## 2b. The figure

![Mean error over the sixteen claims a probabilistic LCA makes, each divided by
its own true level, against the cutoff on a log axis. **Both curves sweep the
same thing -- where the method switches -- and they share their BELOW branch**,
so the vertical gap between them is purely what knowing market share buys on the
materials above the cutoff.

**Orange: KDE with uniform weights above the cutoff, three-parameter lognormal
with uniform weights below. This is the rule a reader can follow.** Blue: the
same, except the materials above the cutoff get their true market shares, which
nobody publishes -- so it is a value of information, not a method.

**Each curve's end points ARE the fixed methods and are labelled there**, which
is why the panel carries no separate reference lines: at a cutoff of 3 every
dataset takes the above-method, and at 10,000 every dataset takes the
below-method. Both curves therefore end at the same point, lognormal with
uniform weights, and that shared end is checked to **0.00e+00** on every run.

**The shaded bands are each curve's own indistinguishable span, in its own
colour and on its own half of the panel** -- orange 40 to 170, blue 50 to 110.
Neither is the 80-to-100 band for when market share starts helping: that is a
per-dataset measurement against the true parent (section 4) and does not live on
this axis.

*If the image does not render:* two shallow U curves on a log x axis from 3 to
10,000 declarations. The orange one runs from 23.92 percent at a cutoff of 3
down to 23.24 around 130 and back up to 23.96 at 10,000. The blue one runs from
22.93 down to 20.25 around 70 to 80 and back up to 23.96, meeting the orange
curve there.

## 3. What the rule buys, claim by claim

At the best cutoff, against the best of the three uniform-weighted methods **on
each claim** -- a harder test than the pooled comparison, because the comparator
changes per claim. Paired cluster bootstrap over pLCA groups; a star marks an
interval clear of zero. **These are PER-UNIT errors -- the error in one
building's answer, not in an average over many** (decision 207).

    the chance of meeting a budget                +6.06  [+5.30, +6.79] *
    a cap: its chance of saving 5 pct             +2.85  [+1.58, +4.00] *
    a material: its 95th percentile               +2.74  [+2.24, +3.23] *
    a cap: how often it binds                     +2.26  [+1.05, +3.40] *
    the total: its mean                           +1.86  [+1.44, +2.31] *
    the total: its 90th percentile                +1.84  [+0.25, +3.41] *
    the total: its standard deviation             +1.42  [+0.11, +2.84] *
    a material: its share at the building's 95th  +1.36  [+0.63, +2.06] *
    a cap: its mean saving                        +0.87  [+0.56, +1.20] *
    a material: its standard deviation            +0.59  [-0.38, +1.59]
    the probability B beats A                     +0.17  [-0.47, +0.78]
    the uncertainty index                         -0.16  [-1.39, +1.01]
    a material: its mean contribution             -0.20  [-0.42, +0.03]
    a material: its chance of being largest       -0.41  [-0.83, +0.00]
    a material: its share of the total            -0.50  [-0.67, -0.33] *

**Fifteen claims, not sixteen.** Using 25 percent less of a material removes
exactly a quarter of its share of the building total, so its error IS that
share's error and the two rows are one claim (decision 186). It is dropped here
and the published counts are 9 of 15 and a median of +1.4 percent.

**The verdict is stable across the band**: median gain +0.87 at a cutoff of 50,
+0.99 at 80, +1.19 at 100, +1.36 at 130, +0.84 at 150.

**So what.** The rule buys accuracy on the tail and intervention claims -- will
this building meet its budget, what will a specification cap deliver -- and
costs a little on the averages, where a lognormal fitted to everything is
already about as good as anything. Worth doing, not transformative.

    Reproduce: outputs/tables/TABLE_MixedPolicyGain.csv, rule=feasible

## 4. Market share: not the problem, too few declarations to use it is

Share of datasets on which a fit that knows the true shares is closer to the
truth than the same fit with every declaration weighted equally:

    declarations       3-9   10-80   81-99  100-999   1000+
    lognormal         32.8    42.5    53.6     64.6    78.7
    kernel            38.5    44.1    52.7     59.0    76.2

**The crossover is in the MEAN, and it is established on the whole corpus.** W1
is bounded below by the distance between two means, so each fit's distance to
the true parent splits into LOCATION -- is the fit aimed at the right
population -- and SHAPE. On all 10,000 datasets, the paired per-dataset
difference in the location term, kernel at each fit's own best bandwidth:

    band      datasets   known minus uniform       t
    3-9          2500        +0.0359            +8.6   knowing shares is WORSE
    10-80        2280        +0.0152            +3.2   worse
    81-99         220        -0.0016            -0.2   indistinguishable
    100-999      2500        -0.0235           -11.1   better
    1000+        2500        -0.0516           -26.7   better

Significantly negative below 80, statistically zero in between, significantly
positive above 100. The three-parameter lognormal, which has no bandwidth at
all, gives the same shape.

**An earlier version of this report claimed the crossing landed exactly at the
cutoff.** It rested on a win-share over 52 datasets with a binomial standard
error of 6.9 points -- a coin flip -- and two seeds disagreed on the sign there.
The statement above replaces it and is stronger.

**So what, and this is the sentence the paper should carry.** It is not that
market share is unhelpful. It is that market share with very few declarations
makes you over-index on a couple of values that may not represent the product
group carrying the weight. The market-weighted mean of nine EPDs where two carry
ninety percent of the weight is arithmetically close to the average of two
numbers and has the standard error of one. Once enough declarations sit inside
the dominant group, knowing the shares is a large help -- four times better on
the mean above a thousand EPDs.

**One precision to keep.** The sample is unrepresentative of the market at every
size; that is the whole reason weighting exists. What changes with size is how
many declarations the dominant products have, so the paper writes "enough
declarations in the products that dominate the market", not "a representative
sample".

    Reproduce: python audits/weighting_location_shape.py --n 10000
               python audits/bandwidth_neff.py --n 2500

## 5. The rule recovers the fit advantage too, which is what the stage was for

                                      mean W1   cost over the per-dataset oracle
    uniform weights, switch at 130     0.2015             133.8 pct
    Lognormal, uniform weights         0.2077             165.5
    KDE, uniform weights               0.2104             145.7

**This is the answer to decision 166**, which opened this stage. That decision
found a fit advantage heavily attenuated by the time it reached a pLCA answer --
a fit-level crossover around a hundred declarations becoming a claim-level one
near a thousand -- and named the mechanism: a probabilistic LCA picks ONE method for all four of
its materials, so one material's advantage is averaged against three neighbours
drawn at random. Letting the method vary by material removes the averaging, and
the rule improves the fit and the claims together rather than one at the
expense of the other.

    Reproduce: outputs/tables/TABLE_MixedPolicyFit.csv

## 6. Where the gain comes from

Pooled relative error by how many of a group's four materials sit above the
cutoff, for the KNOWN-share rule:

    above the cutoff   groups   Lognormal, uniform   KDE, market   the rule
    0 of 4                117          0.3260            0.3687     0.3260
    1 of 4                572          0.2892            0.3161     0.2735
    2 of 4                948          0.2516            0.2465     0.2171
    3 of 4                690          0.2103            0.1658     0.1504
    4 of 4                173          0.1614            0.0843     0.0843

The two ends are exact and are the control: where the rule moves nothing, it
reproduces the fixed method it collapses to. **Everything it buys is in the
middle**, peaking where two of four materials move -- which is also the
commonest composition.

**This split is computed for the known-share rule only.** The feasible rule's
version is not, and the next run should add it.

    Reproduce: outputs/tables/TABLE_MixedPolicyPooled.csv

## 7. Numbers that moved

**In the study's existing tables: none.** Nine of ten pre-existing tables are
content-identical after the re-run and the tenth differs only in its
`written_utc` stamp.

**In this stage's own earlier output, four things moved and all four were
defects:** the claim that no quantity of EPDs gets a uniform-weighted fit below
a floor, contradicted by this stage's own table (decision 222); the claim that
the crossing lands at the cutoff, which rested on 52 datasets (section 4); a
duplicated claim counted twice in the headline (section 3); and the
indistinguishable band, which was a contest among grid points rather than a
difference test and so moved when the grid was filled in (decision 224).

**AND THIS PAPER QUOTES TWO NUMBERS, BOTH RANGES.** 40 to 170 declarations for
the kernel-estimate-against-lognormal split, and 80 to 100 for where knowing
market share stops hurting and starts helping. **No single-declaration threshold
is published** -- not at the claim level and not at the fit level, where the
earlier point estimate carried the same false precision and the measured band
was already much wider. `MIXED_THRESHOLD = 80` is a round reference constant the
code needs in order to name one policy; it is not a result.

## 8. Still open

| Item | |
|---|---|
| **The composition split is computed for the known-share rule only.** The feasible rule's version needs the next notebook run |
| **Every figure except the scorecard and this stage's still carries the retired weighting labels.** Stage 3 |
| **No test compares `flip.FLIP_THRESHOLDS` with the value notebook 3 recomputes**, and decision 175's two-fit rounding rule has not been re-applied since the constants were recalibrated. Stage 3 |

## 9. Inputs, outputs, reproduce

**Read:** the synthetic corpus and the true parents replayed from it;
`TABLE_MethodScores.csv` from notebook 2.

**Written:** thirteen `TABLE_MixedPolicy*` tables,
`CompareUQMethods_FIG_MixedPolicy.png`, and two audit tables,
`TABLE_BandwidthNeff.csv` and `TABLE_WeightingLocationShape.csv`. **Code:**
`src/mixedpolicy.py`, `tests/test_mixedpolicy.py` (36 tests),
`audits/bandwidth_neff.py`, `audits/weighting_location_shape.py`, the swap-draw
hoist in `src/plca.py`, and nine cells at the end of notebook 3. **628 tests
pass, 2 skipped** -- the skips are notebooks 1 and 2 having no `OUT` cell, which
is Stage 3's first task.

    cd notebooks && python -m nbconvert --to notebook --execute \
      --ExecutePreprocessor.kernel_name=compareuq \
      --output-dir=/tmp/nbrun --output=out.ipynb 03_CompareUQ_PerformPLCA.ipynb

About 3 hours 15 minutes, of which the Stage 2j block is about 75.

## 10. What Stage 3 picks up first

1. **Give notebooks 1 and 2 an `OUT` cell AND mark their twelve unmarked
   `savefig` cells.** There are two blockers, not one: the renderer raises on a
   missing setup cell before it ever checks markers, and two tests skip today
   saying exactly that. Clearing both gives seven-second figure rounds.
2. **Add the FEASIBLE rule as a seventh scorecard column** (decision 211), never
   the known-share one. Note that decision 211 was written when the rule won all
   sixteen claims; the feasible rule wins 9 of 15, so `best_method` and `stakes`
   move on some rows and not others.
3. **The caption sweep**: the retired weighting labels, and the corrected
   framings of decisions 212, 219 and 222.
