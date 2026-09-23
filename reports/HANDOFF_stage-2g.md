# HANDOFF stage-2g - Which metric the paper leads with

US spelling throughout, as in every file this project writes.

**HOW TO READ THIS FILE. Its reader does NOT have this repository** -- no
CLAUDE.md, no other report, no table, no figure, no source. Every claim below is
therefore stated in full where it is made, and a trailing `decision N` or
`entry N` is a citation into the project's decision log or its manuscript
discrepancy log, never the substance of the sentence. Where a file has to be
opened, the instruction is addressed to the NEXT CLAUDE CODE SESSION and says
so. The two figures are embedded and their captions carry the numbers, so a
reader whose copy does not render the images loses nothing.

---

# IF YOU READ ONE PAGE, READ THIS ONE

**Stage 2g asks which of the numbers a probabilistic LCA produces the paper
should lead with, and it is the first stage that could answer by measurement
rather than by argument.** Every earlier discussion of the headline metric was
about whether it was STABLE. Since the previous stage ran every probabilistic
LCA a second time against the true distributions the synthetic data was drawn
from, the question becomes whether a fitted model gets the metric RIGHT, and
that is a different and better question.

Each claim below carries the number behind it and a plain-language **so what**
for a reader who builds buildings rather than statistical models.

## 0. The frame: a probabilistic LCA answers five questions

Every result in this study belongs to one of five, and the paper should be
organized by them. In a designer's words:

    what is the building's total embodied carbon?             magnitude
    which materials contribute most, and how often?           attribution
    which materials contribute most to the UNCERTAINTY?       information
    how effective is a reduction strategy?                    action
    is this design better than that one?                      comparison

The second is deliberately two things in one question, and it is the right
grouping: "which material is biggest" and "how often is it biggest" are the
same attribution question asked as a point estimate and as a probability, and
the study reports six numbers that sit somewhere between them.

**That frame is what makes this stage's main finding legible.** "Which material
is the largest contributor" is one of six numbers inside ONE of the five
questions -- not the study's subject, which is how the study has been reading it.
And the uncertainty index is the whole of the third question rather than a
footnote to the second.

**TWO DIFFERENT QUESTIONS, AND THE PAPER NEEDS BOTH.** "How much does the
choice of method cost" is the spread between the best of the six and the worst.
"How good is the answer at all" is the best method's own distance from the
truth. A claim can have a small spread because every method is equally right or
because every method is equally wrong, and only the second number tells them
apart.

**THE UNIT, STATED ONCE, BECAUSE EVERY PERCENTAGE IN THIS SECTION AND IN THE
FIGURE IS THE SAME ONE.** Take the typical gap between what a method says and
what the true distributions say, and divide it by how big the true answer is:

    mean absolute error against the truth  /  the mean TRUE LEVEL of the
                                              same quantity

So 8.0 percent on the building's mean total means the typical method is out by
8 percent of that total. It is a relative error and it is written the same way
for all sixteen claims, which is what lets two of them be compared.

It is a RATIO OF MEANS and not the textbook mean absolute percentage error,
which averages a ratio taken case by case. That form is not usable here: the
true uncertainty index reaches -0.000671 and 2,904 of the 60,000 materials
carry a true value below a hundredth of the mean, so a per-material ratio is
unbounded and sometimes has no sign. Dividing the average error by the average
true level is stable and is defined on every row.

**HOW MUCH THE CHOICE COSTS**, as the gap between the best and worst of the six
on the claim that costs most in each question:

    action        31.0 pct   how often a specification cap binds
    attribution   15.8 pct   a material's chance of being the largest
    magnitude      8.6 pct   the standard deviation of the building total
    information    1.9 pct   the uncertainty index
    comparison     0.9 pct   the probability one design beats another

**AND HOW FAR THE BEST METHOD IS FROM THE TRUTH**, on the worst claim in each
question, in the same units:

    information   43.6 pct   the uncertainty index
    attribution   32.0 pct   a material's chance of being the largest
    magnitude     22.1 pct   the standard deviation of the building total
    action         1.1 pct   a cap's chance of saving 5 pct of the building
    comparison     0.8 pct   the probability one design beats another

**THE TWO ORDERINGS ARE ALMOST REVERSED, AND THAT IS THE SECTION'S POINT.**
Three claims that the first list alone would make look alike:

    a material's chance of being largest   choice costs 15.8   best is off by 32.0
    how often a specification cap binds    choice costs 31.0   best is off by  0.5
    the uncertainty index                  choice costs  1.9   best is off by 43.6

On the first, picking the right method removes about a third of the error and
the rest is there whatever you do. On the second, picking well is almost the
whole problem: the best method is within half a percent and the worst is 31.5
percent out. On the third, picking makes almost no difference and none of the
six is close.

**AND THE UNCERTAINTY INDEX'S 43.6 PERCENT IS NOT SIX METHODS DISAGREEING; IT
IS SIX METHODS WRONG TOGETHER.** Splitting each method's error into the part
all six share and the part that is its own, the shared part is **0.1017 of a
total of about 0.11** -- nine tenths of it. The six methods' per-material
errors correlate 0.66 to 0.99 and all six err in the SAME DIRECTION on **56.9
percent** of materials, against about 3 percent if they were independent. The
mechanism is dataset size: the error is **0.166** for a material estimated from
three to nine declarations and **0.073** above a thousand, and the signed error
runs from **-0.093** to **+0.046** across that range. Every method understates
the variance of a material it has seen nine values of, and because the index is
a variance SHARE that must sum to one, the share the small material loses is
handed to the large ones. It is a property of the DATA, not of the method,
which is why changing method does not help.

> **So what.** For most of what a probabilistic LCA says, choosing a better way
> of modeling uncertainty moves the answer a little closer to the truth rather
> than to it. The one place the choice nearly settles the matter is the
> specification cap, and the one place no choice helps is the question of which
> material drives the uncertainty.

**The two questions a designer acts on most directly -- what will the building
be, and is this design better -- are the two the choice of method affects
least.** What it affects most is how effective a specification cap looks, and
that is driven almost entirely by the two normal fits.

## 1. The number this study HAS led with is the one its methods get most wrong

**THE MEASUREMENT, in plain terms, because it is the statistic the whole stage
rests on.** Every synthetic dataset was drawn from a distribution this study
knows, so the same probabilistic LCA can be run twice on the same random draws:
once with the fitted models and once with those true distributions. The second
run is the right answer. For each number the study reports about a material --
what it contributes, how uncertain that is, its chance of being largest, and so
on -- take the typical gap between the two runs.

**A raw gap is not comparable between one number and another**, because a
contribution is about 1.0 and a chance of being largest is about 0.25. So the
gap is divided by **how much that number genuinely varies from one material to
the next**, which is the signal the number exists to carry. If material A really
contributes 1.3 and material B really contributes 0.8, the difference between
them is what a reader is trying to see; the question is how the error compares
with it.

**THIS SECTION'S DIVISOR IS NOT SECTION 0'S, AND THE TWO MUST NOT BE QUOTED FOR
EACH OTHER.** Section 0 and the figure divide by the true LEVEL of a quantity
and answer "how wrong is this number". This section divides by the SPREAD of
that quantity between materials and answers "can this number tell two materials
apart", which is the only one of the two that can rank one candidate metric
against another -- a metric whose error is smaller than its own
between-material spread is usable and one whose error is larger is not, and no
statement about levels says that. The two are not a fixed multiple of each
other: across the seven numbers below the level runs from 1.17 to 6.57 times
the spread. A figure or a sentence that mixes them is comparing unlike things.

    0.4   the typical error is four tenths of the real spread between
          materials. A reader can still tell materials apart.
    1.0   the error is as large as the whole real spread between materials.
          The number carries no usable information about which is which.

Lower is better, and it is an error against the TRUTH -- not a comparison with
the other methods, which is the separate reading in the figures below.

    number the study reports about a material    best method to worst
    its standard deviation                           0.42 to 0.50
    its 95th percentile                              0.45 to 0.51
    its mean contribution                            0.51 to 0.71
    the uncertainty index                            0.51 to 0.53
    its share when the building is at its 95th       0.61 to 0.69
    its mean share of the building total             0.66 to 0.88
    ITS CHANCE OF BEING THE LARGEST CONTRIBUTOR      0.72 to 1.07

**The study's own headline is last on both ends, and under a normal fit it goes
past 1.0** -- 1.037 with equal weights and 1.074 with market-share weights. So
with a normal fit, the error in a material's chance of being the largest
contributor is larger than the entire real spread of that number across
materials: the ranking it produces carries no information about which material
is which.

**AND IT IS NOT ONE METHOD'S PROBLEM: the chance of being largest is the
worst-recovered of the seven for EVERY ONE of the six methods**, without
exception. Method by method, its recovery error and the next worst metric for
that same method:

    kernel estimate, equal weights          0.733   next worst 0.686
    kernel estimate, market-share weights   0.743   next worst 0.699
    lognormal, equal weights                0.719   next worst 0.658
    lognormal, market-share weights         0.767   next worst 0.699
    normal, equal weights                   1.037   next worst 0.840
    normal, market-share weights            1.074   next worst 0.879

That is what makes this a statement about probabilistic LCA rather than about
one way of doing it. **Whichever method a practitioner uses, the least reliable
number it gives them is the chance that a material is the largest
contributor.**

This is a fourth independent argument for demoting it, and the first one that is
about accuracy. The other three were that it is fragile when four materials
contribute equally, that it carries a 3.67 percent noise floor from an arbitrary
tie-break, and that it is only 9 percent predictable from a material's own data
because it is a property of the GROUP the material sits in.

**AND THE INSTABILITY IS NOT AN ARTIFACT OF THIS STUDY'S CONSTRUCTION, which
is why the paper should report it plainly rather than defensively.** Every
material here is normalized to the same average and given the same use
intensity, which makes a ranking as fragile as it can be made, and that is a
fair objection to the numbers above. But Marsh, Lewis, Hattam and Allen (in
press) find the same thing in a real four-option staircase design: the
top-contributing product changes with which uncertainty characterization
scenario is used. An independent study, on a real element, with real
quantities, sees the ranking move for the same reason. **So this is a property
of ranking near-equal contributors, not a property of synthetic data**, and the
right response is to report the magnitudes and say what the ranking is worth,
not to defend the ranking.

> **So what.** "There is a 30 percent chance this material is your biggest
> source of carbon" is the least trustworthy sentence a probabilistic LCA
> produces. How much the material contributes, and how uncertain that is, are
> both recovered far better. Lead with those. And a published study of a real
> staircase saw the same thing happen, so this is not an artifact of the
> synthetic test.

## 1b. What to lead with instead: how much, and how uncertain

A probabilistic LCA answers two different questions about every material -- how
much it contributes, and how confident you can be in that -- and the study has
been leading with a third that conflates them into a single ordering. The
replacement is the pair:

**A material's estimated contribution**, which is the magnitude, recovered at
0.51 to 0.71 of its own between-material spread; and **the spread of that
contribution**, which is the confidence, recovered at 0.42 to 0.50 and the best
of the seven candidates. Beside them, the **uncertainty index** answers the
third question a designer actually acts on -- where to collect better data --
and is the output the choice of method affects least.

**The chance of being the largest contributor stays in the paper as one
statement among five, quoted with its noise floor**, because it is what the
manuscript currently reports and because readers will look for it. It is not the
frame.

**Why the pair and not a single number.** The two are not substitutes, and no
one method is best at both: the kernel estimate with market-share weights is the
best of the six at a material's standard deviation, and the lognormal with
market-share weights is the best at its mean contribution. A paper that reports only one of the two
gives a reader no way to tell a material that is big from one that is uncertain,
and those call for different actions: the first is a design problem and the
second is a data-collection problem.

> **So what.** Report how much each material contributes and how uncertain that
> is, as two numbers, and report which material's uncertainty is worth reducing.
> "Which material is biggest" is a summary of the first two and is the least
> reliable thing on the page.

## 2. No UQ method is best for every claim, and the best one changes

"Way of describing uncertainty" was the wrong phrase and is dropped: what the
study compares is six **UQ methods** -- three shapes (kernel estimate,
three-parameter lognormal, normal) crossed with two weighting schemes (equal,
market-share). The claim is about those six, not about how a result is written
up.

Sixteen claims a probabilistic LCA makes, scored for all six against the truth
on one definition, is the figure below. Three things come out of it.

**No single method is best everywhere.** The six differ measurably on 15 of the
16 claims, and on those 15 four different methods are best on something: the
three-parameter lognormal with equal weights takes 5, the same lognormal with
Dirichlet-drawn shares 5, the kernel estimate with Dirichlet shares 3, and the
kernel estimate with equal weights 2. **Neither normal fit is ever first.** The
one claim on which the six do not differ is what a quantity reduction saves,
where all six and the truth agree to four decimal places.

**THAT COUNT IS NOT A RANKING OF METHODS AND MUST NOT BE QUOTED AS ONE.** It is
an average over every dataset size, and the ordering INVERTS with size. Mean
error across the seven per-material claims, each as a percentage of its own
true level:

    n           KDE eq  KDE Dir  Logn eq  Logn Dir  Norm eq  Norm Dir
    3-9           40.4    43.6     40.1     45.5      41.8     45.6
    10-99         25.7    26.8     23.7     25.9      27.5     28.3
    100-999       18.1    15.8     18.1     15.3      22.1     20.5
    1000+         16.3    11.0     17.2     12.6      20.8     17.9

The lognormal under equal weights is closest in the two bands below 100
declarations, the lognormal under Dirichlet shares at 100 to 999, and the
KERNEL ESTIMATE under Dirichlet shares above 1,000, at **11.0 against the
equal-weighted lognormal's 17.2**. Both axes turn over: equal weights win every
band below 100 and Dirichlet shares win every band above.

**The pooled count leans to the lognormal because the corpus puts 2,500
datasets in each of four size bands**, so half of every probabilistic LCA sits
below 100 declarations. That allocation is an experimental design choice.
Reweighting to the real size mix of the 147 EC3 categories -- 14 / 54 / 26 / 6
percent -- moves it FURTHER toward the lognormal, because two thirds of real
categories hold fewer than 100 declarations.

**How much the choice costs varies by a factor of about thirty-five across the
claims**, from 31.0 percent of the true value on how often a specification cap
binds, down to 0.9 percent on whether one design beats another, and to nothing
at all on what a quantity reduction saves.

**And on most claims most of the error is there whatever you choose.** On a
material's chance of being the largest contributor the worst method is 47.7
percent out and the best is still 32.0 percent out, so 32.0 of those 47.7
points are there under every method and 15.8 is what the choice adds. On how
often a specification cap binds the split is the other way round: the worst
method is 31.5 percent out and the best is 0.5, so almost the whole error IS
the choice. That contrast is what the figure's right-hand bar exists to show
against its cells.

> **So what.** There is no method you can adopt once and stop thinking about,
> and for most of what a probabilistic LCA says, switching method moves you a
> little closer to the truth rather than to it. Two exceptions are worth
> knowing. If you want to know how often a specification cap will actually
> bite, the method you pick decides the answer and a normal distribution gets
> it badly wrong. And how many declarations you hold decides which of the other
> two to reach for: a three-parameter lognormal on a thin category, a kernel
> estimate on a well-populated one.

## 3. "Never use a normal distribution" is about three of the five questions, not about everything

Neither normal fit is ever the best method on any of the sixteen claims. How
far the better of the two normal fits sits behind the best method, in the same
percent-of-the-true-value units as everywhere else:

    how often a specification cap binds          26.98   normal is worst
    a cap's chance of saving 5 pct               24.85   normal is worst
    a material's chance of being largest         14.11   normal is worst
    a cap's mean saving                          10.97   normal is worst
    the standard deviation of the total           7.40   normal is worst
    the standard deviation of a contribution      4.91   normal is worst
    a material's mean contribution                4.22   normal is worst
    the chance of meeting a budget                3.75   normal is worst
    a material's share of the total               2.76   normal is worst
    the mean of the total                         2.40   normal is worst
    a material's 95th percentile                  0.82   NOT worst
    the 90th percentile of the total              0.74   normal is worst
    the probability one design beats another      0.62   normal is worst
    a material's share at the building's 95th     0.63   NOT worst
    the uncertainty index                         0.42   NOT worst
    what using 25 pct less saves                  0.00   NOT worst

**The penalty spans a factor of sixty, and the top of the list is not
attribution.** It is the specification cap, where a normal fit says the cap
binds 27 to 31 percent of the time more wrongly than the lognormal does, and
then attribution. **On the bottom three claims it is within one percentage
point of the best method and is not the worst of the six**: a material's 95th
percentile, its share when the building is at its 95th, and the uncertainty
index.

**What survives every claim is the BIAS, and that is the part that matters for a
whole building.** On the three numbers where a signed error means something --
a material's mean contribution, its 95th percentile, its standard deviation --
the normal is the most biased of the six on all three, by **+4.8, -10.5 and
-28.5 percent** of the true level, against the kernel estimate's +0.3, -4.5 and
-14.1. Bias adds across the materials of a building while random error cancels,
so a method that leans is a worse problem at building scale than one that is
merely imprecise.

**One warning about reading that.** On a share or a rank frequency the signed
error is identically zero for every method, because the four values sum to one.
That is arithmetic, not evidence of unbiasedness.

> **So what.** The advice to stop fitting normal curves stands, and it is worth
> saying where it bites hardest: for judging how much a specification cap will
> deliver, for saying which material dominates, and for anything a building
> total is added up from. For describing the high end of a single material, or
> for deciding where to collect better data, a normal is about as accurate as
> anything else -- it just leans the same way every time.

## 4. The five questions, with the numbers, in the order the results section should take

The order below is NOT the order of section 0, and the difference is
deliberate: section 0 is how a reader thinks about a probabilistic LCA, and this
is how the results read best -- the decision first, because the answer there is
a null and a null is the strongest thing this stage has.

1. **The comparison, which leads, because it is the decision a designer makes
   and the answer is a null.** Over **2,500** pairs of designs differing in one
   material, the choice of method changes the stated probability that the
   substitution is an improvement by at most **0.015**, and every method lands
   within **0.026** of the truth. At a claimed 5 percent saving the truth is
   **0.630** and the six methods span 0.638 to 0.647. **This was 800 pairs
   until the author asked why the stage's headline null used a tenth of the
   sample every other truth-run result uses; at 800 the interval on that
   0.015 was as wide as the number itself.**
2. **The safe-lead rule.** The chance that the choice of method changes which
   material leads crosses 1 percent at a top-two contribution ratio of **2.13**
   (95 percent interval 2.09 to 2.17). The one real building element available,
   the Concrete-Precast staircase of Marsh, Lewis, Hattam and Allen (in press),
   sits at **1.02**.
3. **The building total and the budget statement.** Every method understates the
   total's 90th percentile, by **0.087 to 0.356** on a building whose total
   averages about 4.0, and at a budget the truth meets 90.0 percent of the time
   the six report **86.8 to 91.3** percent.
4. **The specification result, where the choice costs most.** Against a true
   mean saving of **5.39 percent** of the building from capping a material at
   the 75th percentile of the declarations held, the six report **4.88 to 6.19**
   percent; asked for the chance of achieving at least a 5 percent saving, the
   truth is **23.2** percent and the six span **22.9 to 30.5**.
   **And beside it the contrast that explains the whole stage:** a QUANTITY
   reduction delivers **0.0625** of the building under every method and under
   the truth, identical to four decimal places, because it is a deterministic
   fraction of a material's own contribution and no distributional assumption
   enters at all.
5. **Where the uncertainty sits**, which is claim 5 below.

> **So what.** Choosing between two designs is safe under any of these methods.
> Saying which material is biggest is safe only if it leads the next by about a
> factor of two, and real designs do not always. Saying what a
> low-carbon-specification policy will buy you is where the method you picked
> shows up in the answer.

## 5. The steadiest number a probabilistic LCA produces deserves promoting, and no method gets it right

**A correction first, because two earlier stages got this wrong and this one
repeated it.** The claim originates three stages back and was carried forward
twice: that the uncertainty index "appears in no table, figure or section" and
is "reported nowhere". **That is false.** The
manuscript reports it in two panels of Figure 5, defines it in the supplement,
and draws a conclusion from it in as many words -- that the methods give similar
uncertainty indices. What is true is narrower: it is reported as a secondary
observation about how far apart the methods are, rather than as one of the five
questions a probabilistic LCA answers. **The recommendation is to promote it,
not to introduce it.**

The uncertainty index -- which material's uncertainty drives the uncertainty in
the whole building -- has the lowest disagreement between methods of any main
output, **0.5035** against **1.042** for a material's chance of being largest.
Asked which material's uncertainty dominates, the best method names the truth's
answer **58.4 percent** of the time against a one-in-four chance level, the
highest of any candidate -- though effectively tied with the spread of a
material's contribution at 58.4 percent as well, and against **52.5** percent
for the chance of being largest and **49.1** for the estimated contribution.

**And every one of the six methods is out by about half the metric's own spread
between materials**, 0.508 to 0.531, a span of only 4.4 percent from best to
worst. So the choice of method genuinely does not matter for it, and no method
gets it right.

**That pairing is the general lesson of this stage.** A number the methods agree
about and are all wrong about is the one thing a study must not present as
reliable, and the two statistics are reported side by side so that case is
visible.

It is also the fourth carbon-reduction strategy under another name: three
strategies reduce the expected impact -- use less, specify better, substitute --
and this one reduces the VARIANCE of the answer, which is exactly what obtaining
a supplier-specific declaration buys.

> **So what.** "Where should I spend my next hour of data collection" is both
> the most useful thing a probabilistic LCA tells a designer and the answer
> least affected by how the uncertainty was modeled. It should be in the paper.
> It should also carry the warning that every method is roughly equally
> imprecise about it.

## 6. The goodness-of-fit score cannot see how far out a bad tail goes

A distance between two cumulative curves charges for how much mass a model
misplaces; a Monte Carlo simulation samples from the model and is wrecked by how
FAR out that mass sits. A model with a thin enormous tail would therefore score
well and dominate any simulation it entered.

Measured, by moving a thousandth of one fitted model's mass out to each of 25
distances from 1 to 3,000 times the dataset's mean, holding the other three
models, the random draws and the group: **the evaluation grid ends at 9.9 times
the mean, and at all 16 distances beyond it the score taken over that grid alone
is the same number -- 0.262571 -- to the last digit**, against an uncontaminated
0.254474. It cannot tell 15 times the mean from 3,000. With the correction the
previous-but-one stage added, which integrates the model's remaining tail
analytically, the same 16 distances run from **0.2729 to 3.268**, a factor of
twelve.

**And the metrics split by whether they have a ceiling.** Relative change over
those same 16 distances, from the first beyond the grid to the last:

    spread of a material's contribution      0.203   ->  104.8
    the uncertainty index                    0.215   ->    1.46
    its estimated contribution               0.013   ->    1.99
    its share at the building's 95th pct     0.00823 ->    0.00823
    its mean share of the total              0.00203 ->    0.00246
    its chance of being largest              0.00129 ->    0.00129

A share and a rank frequency saturate: once a material's draw is enormous it
holds the whole share and takes first place, and making it two hundred times
more enormous changes neither to the last digit. A mean, a standard deviation
and a variance share have no such ceiling; the spread of a material's
contribution moves by a factor of 517 over the same range.

**This creates a tension the paper has to state rather than resolve.** The
metrics that recover the truth best are levels, and levels are exactly what a
thin far tail wrecks; the ones that are immune are shares, and they recover
worse. What makes the levels safe to report is that the study's own criterion
now charges for the thing that wrecks them, which it did not two stages ago.

**AND IT IS NOT HYPOTHETICAL, WHICH HAD TO BE CHECKED RATHER THAN ASSUMED.**
The fitted model's own spread over the data's is near 1.0 in the median for
every one of the six methods on both halves of the study, so the typical fit is
fine -- and **0.25 percent of fits exceed five times the data's spread**. The
worst single fits reach **73.7**, 50.1 and 40.6 times, and all three are
EQUAL-WEIGHTED fits to small datasets; their market-share-weighted twins top out
at 5.0, 1.4 and 1.0. Two guards already keep this out of the results: a bound on
the lognormal's third parameter at fitting time, and the tail correction in the
criterion. Measured, the part of the score lying beyond the grid is **exactly
zero** for the kernel estimate and for the normal, which put no mass there, and
averages 0.00006 and 0.00009 for the two lognormals.

**TRUNCATING EACH FITTED MODEL AT THE TOP WOULD REMOVE THE FAILURE MODE
OUTRIGHT, and this stage states that rather than implementing it.** Every model
here is already truncated BELOW at zero, because a negative emission coefficient
is not admissible and that bound needs no argument. An upper bound has no
equally external anchor -- the physical ceiling this study applies to the raw
declarations is in their own units, and every dataset here is rescaled to an
average of 1.0 -- so choosing one is a modeling decision with numbers attached,
which belongs in a sweep. It is handed to the next stage.

> **So what.** A curve that looks like a good fit can still hide a rare,
> enormous value, and the simulation you run afterwards will be dominated by it.
> The check is worth running on any goodness-of-fit number before trusting it.

## 7. A metric with a comment saying its numbers look wrong, settled

One reported quantity -- which material is the best one to cap, as a frequency
over simulation iterations -- carried a fudge factor and an inline comment
conceding that "percentages look off because not all reduction strategies apply
in all scenarios". The comment named the right problem; the fudge factor was the
wrong fix.

**The correct denominator is the iterations in which capping something helps at
all.** In an iteration where no material's cap binds there is no best material
to cap, and counting those iterations against all four made the column depend on
how often the strategy applies rather than on which material is the right one.
The frequencies now sum to **exactly 1.0** across the four materials, like every
other such frequency in the study.

**And how often the strategy applies is now reported rather than assumed, which
is where the signal was.** The share of iterations in which any cap binds runs
from **0.727** under a lognormal with equal weights to **0.838** under a normal
with equal weights -- a method that puts more mass above the cap finds the cap
binding more often. The old fudge factor forced that number to 0.25 for every
material and every method.

**BUT APPLICABILITY IS A DIAGNOSTIC, NOT THE DELIVERABLE, and the paper should
lead with the effectiveness.** What a reader wants from a specification cap is
what it saves and how likely it is to: against a true mean saving of **5.39
percent** of the building the six report 4.88 to 6.19, and asked for the chance
of achieving at least 5 percent the truth is **23.2** percent while the six span
**22.9 to 30.5**. That is the claim, and it is the one quoted in the cost table
at the top of this file.

**Applicability earns its place as the MECHANISM behind that spread**, and it
can now be scored against the truth, which the constant divisor made impossible.
The true distributions say a cap helps in **0.279** of iterations. The lognormal
with equal weights says 0.277 and with market-share weights 0.282 -- both
indistinguishable from the truth, their bootstrap intervals straddling zero
error. The kernel estimate is high by 0.012 and 0.020, and **the normal is high
by 0.088, which is 31 percent too often** -- which is why it also overstates
what the strategy delivers.

That ordering is the reverse of the one on a material's chance of being largest,
where the kernel estimate with market-share weights leads. The lognormal's
strength is the shape of the upper tail, which is what a cap acts on; the kernel
estimate's is following the body of the data.

> **So what.** "Cap the carbon of your worst-performing material" is worth more
> under some ways of modeling uncertainty than others, because they disagree
> about how often any product would actually exceed the cap. That disagreement
> was previously invisible by construction.

---

## The figure

![Every claim a probabilistic LCA makes, scored for all six UQ methods against the truth, and the same comparison split by dataset size](../outputs/figures/CompareUQMethods_FIG_ClaimScorecard.png)

**Figure: under the best of the six methods a probabilistic LCA is right to 0.8
percent on the design comparison and wrong by 32.0 percent on which material
leads -- and the pooled count in the upper panel is not a ranking of methods,
because the ordering inverts with dataset size.**

**EVERY CELL OF BOTH PANELS IS THE SAME QUANTITY**, which is what makes two
rows, and the two panels, comparable: that method's mean absolute error against
the truth, as a percentage of the mean true level of the thing being claimed. A
black box marks the method closest to the truth in each row. **"Dirichlet
shares" is what the study has been calling "Variable"**: the market shares were
drawn from a flat Dirichlet because nobody publishes them. "Equal" means every
declaration counts the same.

**UPPER PANEL, what the cells say by question.**

- **The building total.** The best method is 8.0 percent out on the total's
  mean, 10.7 on its 90th percentile, 5.5 on the chance of meeting a budget and
  **22.1 on its standard deviation**, the magnitude claim every method gets
  worst.
- **Attribution.** The best method is **32.0 percent out on a material's chance
  of being the largest contributor** and the worst is 47.7. Its mean
  contribution is better served, 11.6 to 16.2, and its share of the total
  better still, 10.0 to 13.4.
- **Information.** The uncertainty index is the worst-recovered claim on the
  figure, **43.6 to 45.5 percent**, with only 1.9 points between best and
  worst. Nine tenths of that error is common to all six methods and is the
  error of estimating a variance from few declarations, so no choice of method
  repairs it.
- **Action.** The widest row, and it is a split between families rather than a
  spread: the two lognormals are 0.5 and 1.0 percent out on how often a
  specification cap binds, the two kernel estimates 4.2 and 7.2, and the two
  normal fits **27.5 and 31.5**. And **on what using 25 percent less of a
  material saves, every method is exactly right**, because that intervention is
  a deterministic fraction of the material's own contribution and no
  distributional assumption enters.
- **Comparison.** The best method is **0.8 percent** out and the worst 1.7.

**The right-hand bar is a different question from the cells, and reading only
one misleads.** It is the worst method's error minus the best method's, so it
is what the CHOICE of method costs, where the cells say how good the answer is
at all. On how often a specification cap binds the choice costs 31.0 and the
best method is 0.5 out: picking well is nearly the whole problem. On a
material's chance of being largest the choice costs 15.8 and the best method is
32.0 out: picking well is about a third of it. On the uncertainty index the
choice costs 1.9 and every method is about 44 out: picking makes no difference
and none of them is close.

**LOWER PANEL: the same seven per-material claims, pooled, split by the
material's own dataset size.** The lognormal under equal weights is closest
below 100 declarations, the lognormal under Dirichlet shares from 100 to 999,
and the kernel estimate under Dirichlet shares above 1,000, at **11.0 against
the equal-weighted lognormal's 17.2**. **Both axes turn over**: equal weights
win every band below 100 declarations and Dirichlet shares win every band
above.

**Why the panel is there rather than a caption warning.** Counting boxes in the
upper panel gives the lognormal 10 rows to the kernel estimate's 5, which reads
as a verdict between the families. It is an average over a corpus that puts
2,500 datasets in each of four size bands, so half of every probabilistic LCA
sits below 100 declarations -- and reweighting to the real size mix of the 147
EC3 categories moves the count further toward the lognormal still, because two
thirds of real categories hold fewer than 100 declarations. A caption telling
the reader not to count would have asked them to take that on trust.

**TWO OTHER FIGURES WERE BUILT AND CUT IN THIS STAGE.**

The first showed each candidate metric's recovery error as a best-to-worst
range with the metric names on one axis. It hid which method was which, its row
labels were vague, and the scorecard says everything it said and more. Its one
unique content -- that a material's chance of being largest is the only number
whose error exceeds the between-material spread it has to reveal -- is section
1 of this handoff.

The second showed what a thin far tail does to a goodness-of-fit score. It was
cut because it is a stress test rather than an observation: across 60,000 fits
on both halves of the study the mean charge for mass beyond the evaluation grid
is 0.0000 to 0.0001, a fraction of a percent of fits exceed five times the
data's own spread because the guard on the lognormal threshold already bounds
them, and truncating each fitted model at a plausible multiple of the largest
observed value would remove the failure mode outright -- which is Stage 2h's to
sweep. **The finding itself survives in full and is the body section titled "the
goodness-of-fit score cannot see how far out a bad tail goes", with every number
it needs.** Its title was also unreadable: "W1 stops charging once the tail
leaves its grid" used "charging" to mean "adding to the score", and a title that
needs its own vocabulary explained is not a title.

---

## 1. Stage and branch

| | |
|---|---|
| **Stage** | 2g, the metric set |
| **Branch** | `stage-2g-metric` |
| **Branched from** | `dff34fb` on branch `stage-2f-multivariate`, working tree clean |

Commits, in order: the source module and its tests; the notebook changes; the
decisions and the mechanics documentation; the manuscript discrepancy entries;
the run; then the author's review, in four rounds -- redrawing the figures and
cutting two of them; moving the scorecard from each method's excess over the
best method to its total distance from the truth; putting every scorecard row
on one definition of that distance; and finally adding the size-band panel,
renaming the weighting schemes for display, and raising the design comparison
from 800 pairs to 2,500. The notebook was re-run after each round that changed
a table.

---

## 2. What was asked

Report the five statements a probabilistic LCA makes, in a specified order, with
the design comparison leading. Add the uncertainty index, which is the steadiest
output measured and appears in no table, figure or section, and evaluate it
against the run-against-the-truth like any other candidate. Finish the fudge
factor on the cap rank frequencies: confirm the new definition is the one the
paper wants and that nothing downstream still assumes the old one. Judge every
candidate headline metric by how closely it reproduces the run against the true
distributions, which is the comparison that arrives from the previous stage and
changes the question from "is this metric sensitive" to "does this metric
recover the right answer". Add magnitude-based companions -- each dataset's mean
share of the total and its share at the 95th percentile of the total -- and
check whether the conclusions hold under those as well as under the rank metric.
Build a win-share view. Check any recommended metric against the failure mode in
which a model scores well while carrying a thin enormous tail. And fix the one
notebook cell still explaining probabilistic LCA outcomes with an evaluation
target that was retired three stages ago.

---

## 3. What was done

**One new source module with 23 tests.** It holds the recovery statistic and the
argmax agreement, the corrected normalization for the cap rank frequencies, and
the tail stress test with its contaminated-model wrapper. The new magnitude
companion lives beside the other outputs in the probabilistic LCA module,
because the function that computes every output has to compute it too.

**Fifteen new cells at the end of the third notebook**, each one call into that
module, plus four edits inside existing cells: the two magnitude companions, the
corrected cap normalization, and the retired target replaced.

**ONE NEAR-MISS WORTH RECORDING, because it is the failure the project has a
rule against.** The scratch script used to iterate the figures wrote its
derived tables into a scratch directory whose entries are symlinks to the real
results, so that the figures read production data while nothing is written to
it. Writing a file through one of those symlinks follows it, and one derived
table therefore landed in the real results directory and replaced the
notebook's own. It was caught by a timestamp within the hour, restored from
version control, and the scratch script now refuses to write to any path that
resolves inside the results directory. **Nothing reached a commit and no
published number moved**, and the mechanism is the same one the project's rule
"the results directory is written by the notebooks and by nothing else" exists
to prevent -- the rule held for the notebooks and said nothing about a scratch
script's symlinks.

**A SECOND PASS AFTER THE AUTHOR'S REVIEW added five things.** The claim
scorecard against all six methods, which is the figure the review asked for. A
test of whether the win-share leader is a leader or a tie, which found that one
of the three leaders the first pass named was noise. The cap's applicability
scored against the truth, which the old constant divisor could not have asked.
A check on whether any model this study actually fits has the runaway tail the
stress test simulates. And a continuous distance sweep for that stress test,
replacing three round decades that drew as three points and could not show
WHERE the criterion goes blind.

**A THIRD PASS PUT EVERY SCORECARD ROW ON ONE DEFINITION and cut a figure.**
The first scorecard divided seven of its seventeen rows by the spread of the
true value between materials and the other ten by the true level, then drew
both on one colour scale as percentages. Those are two different statistics and
their ratio is not a constant, so the rows were not comparable -- not across
the five questions and not even within the attribution block, where the level
runs from 1.17 to 6.57 times the spread. Every row now divides by the true
LEVEL. `total_w1` was dropped with that change, because it is a distance whose
true value is zero and has no level to be a percentage of, so the scorecard is
sixteen claims. The spread version is kept in its own table, where it does the
job it is right for: ranking one candidate METRIC against another by whether it
can tell two materials apart. The tail figure was cut in the same pass and its
finding is a paragraph beside the surviving figure.

**The whole test suite is 566 tests, 564 passing and 2 skipped**, including the
eight regression fixtures that pin the dataset characteristics and all six
goodness-of-fit scores. That is the check that the fitting, the corpus and the
empirical arm were not touched. One test stopped skipping rather than starting
to pass: the figure renderer now accepts notebook 3.

**The third notebook was run end to end four times**, once to produce the
tables, once with the figure cells added at the end, once after the denominator
change and once after the design comparison's sample size was raised. **Every run reproduced every table exactly** except the two
the change was meant to touch: the uncompressed tables are byte-identical and
the compressed ones are identical once decompressed, differing only in the
timestamp the compression format embeds in its own header. The recovery table
gained four columns with every pre-existing column bit-identical, and the
scorecard table is the one artifact whose values are intended to move.

---

## 4. Numbers that moved

**One change moved a committed number, and it is a change of definition rather
than of substance.**

**The cap rank frequencies.** Only the four `capecc_rank_*` columns of the
60,000-row results table; **36 of the 40 shared columns are bit-identical** to
the previous run, which is the check that the fitting, the common random numbers
and everything else were untouched. The rank-1 column's mean goes from
**0.1929 to 0.2500**, which is exactly one over four and is the partition
property the correction restores; rank 2 from 0.0922 to 0.1167, rank 3 from
0.0238 to 0.0294, rank 4 from 0.00255 to 0.00306.

**Three columns were added and none of them moved anything.** A material's share
of the building total at the total's 95th percentile, the share of iterations in
which any cap binds, and the share in which this material's own cap binds.
Adding an output consumes no randomness; the safe-lead crossing of **2.132671**
is identical to the last digit across the two runs, and a test recomputes every
other output from the same draws to assert nothing else changed.

**One second-order movement, and it is bootstrap noise on an interval rather
than on an estimate.** The table reporting each method's error against the truth
now covers seven metrics where it covered five, and its confidence intervals
come from a resampling stream shared across the metrics in order, so adding two
shifts the draws the later ones get. **Every point estimate in that table is
unchanged**; the interval BOUNDS move by at most **0.0011**, on intervals
themselves about 0.004 wide.

**Everything else is unchanged and that was checked rather than assumed.** The
run against the true distributions, the design comparison, the flip calibration,
the sweep over group size and material use intensity, and the eight regression
fixtures all reproduce. Three figures change and all three for the same reason:
they draw every result column, so they gain panels for the new ones and redraw
the corrected cap panels.

**AND THE LATE DENOMINATOR CHANGE MOVED NOTHING BUT THE SCORECARD ITSELF.**
Putting every scorecard row on the true-level divisor rewrites one table and
one figure, and a fourth full run of the third notebook confirmed the rest is
untouched: the nine compressed tables that appear to differ are identical once
decompressed, the 130 other tables are byte-identical, and the recovery table
gained four columns -- the true level, and the error divided by it with its two
interval bounds -- with **every pre-existing column bit-identical**, so the
spread-based statistic every earlier section of this handoff quotes did not
move by a digit.

**AND A THIRD RUN, FOR THE DESIGN COMPARISON'S SAMPLE SIZE.** Raising it from
800 pairs to 2,500 moved that experiment and nothing else: **every other table
is byte-identical or identical once decompressed**, and the only files that
changed are the design-comparison rows, its summary, the five-statement table
that quotes it, the scorecard's one comparison row and the figure. The truth
it is measured against moves within its old interval -- 0.5051 to 0.5017 at a
claimed nought percent saving, 0.6292 to 0.6304 at five percent -- and **the
spread across the six methods tightens as intended**, 0.0198 to 0.0146 at a
claimed 10 percent saving and 0.0119 to 0.0088 at five.

**One best-method box changed with it, and that is the sample size doing its
job.** On the design comparison all six methods sit within 0.9 points of each
other, and at 800 pairs the closest was the kernel estimate with Dirichlet
shares at 0.53; at 2,500 it is the kernel estimate with equal weights at 0.81,
with Dirichlet shares second at 0.88. The best-method counts over the 15 claims
where the six differ therefore go from 5 / 5 / 4 / 1 to **5 / 5 / 3 / 2** across
the equal-weighted lognormal, the Dirichlet lognormal, the Dirichlet kernel
estimate and the equal-weighted kernel estimate. **A "best method" on a row
whose six values span less than a point was noise, and the larger sample is
what shows it.**

**What the scorecard's own numbers become, because they are quoted in section
0 and they did move.** The claim on which the choice of method costs most is no
longer a material's chance of being largest at 35.5 percent of the
between-material spread; it is **how often a specification cap binds, at 31.0
percent of the true value**, with the chance of being largest second at 15.8.
The error the best method still makes is largest on the uncertainty index at
**43.6 percent**, where the old divisor made it 50.8 percent of a spread. The
best-method counts change by one, because dropping `total_w1` removed a claim
the lognormal with equal weights had won: it is now 5, 5, 4 and 1 across the
two lognormals and the two kernel estimates over the 15 claims on which the six
differ. **No conclusion of this stage reverses**; what changes is which
question is named as the most expensive to get wrong, and the answer is now
the specification cap rather than attribution.

---

## 5. What is still open

### Owned by a later stage

| Item | Owner |
|---|---|
| **THE WEIGHT MODEL, carried forward from the previous stage and still the largest open item.** The two halves of the study draw market-share weights by different rules -- a flat draw over individual declarations on the real categories, weights attached to the humps of the distribution on the synthetic ones -- so the weights are correlated with the carbon coefficients on one and independent of them on the other, which is the dimension the paper is built on. Measured decay with category size: **-0.397 on the real categories against -0.167 on the synthetic**, and above a thousand declarations the median effect is **0.0049 real against 0.0501 synthetic**, a factor of ten. **Nothing in this stage depends on it and no number here moved because of it.** The fix is one rule on both halves with a swept coherence parameter, controlling separately for how concentrated the shares are | 2h, first item |
| The profile-likelihood guard sweep. **This stage adds a binding constraint on it:** the guard against a runaway tail is the only thing making the level metrics safe to report, so the tail term must stay in force and the fitted-model spread ratio must be reported at every value swept | 2h |
| Drawing the sizes of the distribution's humps from a flat draw rather than at concentration 10; multiple weight realizations; the deduplicated variant; **and the pedigree matrix**, which is what connects this paper to the practice most readers use | 2h |
| **THE MANUSCRIPT OWES A PARAGRAPH SAYING WHY THESE THREE FAMILIES AND NOT THE OTHERS**, and it is owed because the others are common rather than obscure. The lognormal is ecoinvent's default and is what a pedigree matrix produces, since a geometric standard deviation IS a lognormal parameterization; the normal is what a great deal of practice uses and is wrong for a strictly positive right-skewed quantity; a kernel estimate is the flexible alternative under test. Gamma was compared and is indistinguishable from the three-parameter lognormal out of sample. **Uniform, triangular and beta are excluded because of what they are for, not because of how they would score:** this paper compares ways of turning a SET of declarations into a distribution, and a uniform is specified from two numbers rather than fitted -- its maximum likelihood fit to n values is exactly the smallest and largest of them. Beta is for bounded quantities such as efficiencies, which an emission coefficient is not. Without that paragraph a reader will assume the three were chosen for convenience | manuscript |
| **A UNIFORM AND A TRIANGULAR DISTRIBUTION, in the same arm as the pedigree matrix and for the same reason.** The author asked whether other shapes are worth comparing. They are not competitors HERE: this paper compares ways of turning a set of declarations into a distribution, and a uniform is not fitted to a dataset -- its maximum likelihood fit is exactly the smallest and largest value, discarding everything between -- so including it would be a straw man, and a straw man that flatters this paper's own method. But a uniform and a triangular are exactly what a practitioner reaches for when there is NO dataset, which is the situation the pedigree matrix is built for, and the yardstick that stage uses does not care how a model was built. Gamma is already settled: out of sample on the real categories it is indistinguishable from the three-parameter lognormal | 2h |
| **AN OPTIONAL UPPER TRUNCATION OF EACH FITTED MODEL**, which would remove the thin-far-tail failure mode outright at the cost of one more assumption. Not implemented here because the lower bound at zero is external and needs no argument while an upper one does not have that anchor | 2h |
| Every figure brought to the style guide; the figure manifest; the older figures still carry a Unicode minus. **The two figures added here follow the guide and pass its own clash detector** | 3 |
| A real-building anchor, if citing the staircase paper is not enough | 2i, optional |
| An industry-average declaration as a direct estimate of the market-weighted mean | unowned |

### Opened here

| Item | |
|---|---|
| **The corpus's characteristic list omits the modality measure the paper should report.** The label file names it and the stored characteristics carry it, but the list that hands characteristics to the third notebook does not, so that notebook's exploratory correlation scan cannot see it. Adding it would change the shape of a table that is also a regression fixture and would need the second notebook rerun and the fixture re-frozen. **Nothing depends on it**: the analysis that uses that measure lives in the fourth notebook and reads the stored characteristics directly. Owned by whichever stage next reruns the second notebook |
| **The tension between accuracy and tail-robustness is stated and not resolved.** The metrics that recover the truth best are levels and levels are what a far tail wrecks; the immune ones are shares and they recover worse. The paper should say so; there is no measurement that settles it |
| **The reports directory holds five handoffs and that is DELIBERATE, by the author's decision of 2026-09-22: "I'm fine holding four handoffs in one spot ... Seems fine to have previous context."** The project brief's rule that only the current stage's handoff is kept is therefore relaxed for these. **A session starting from here should read all of them** -- stages 2c, 2d, 2e, 2f and this one -- rather than assuming this file is the only record. Nothing outstanding lives only in them: the decision log and the discrepancy log still carry every open item, which is what the rule was protecting |
| **British spellings survive in files this stage did not write**, which the author caught in this one. Fixed here and in the decision log; still present in the figure style guide (which uses "colour" throughout), in decision-log entries from earlier stages, and in six cells of the third notebook that earlier stages wrote. Not swept, because it would put unrelated diffs from four stages into this one. Owned by the deposit tidy-up |

### Closed here

The fudge factor on the cap rank frequencies, open since Stage 0 as manuscript
discrepancy entry 7. The last use of the retired evaluation target. The
magnitude-based companions and the sensitivity of the headline rank frequency.
The win-share view. And the tail failure mode, checked rather than assumed.

### Known and accepted

The recovery statistic is measured against the market-weighted true
distribution, which exists only on the synthetic half of the study; the real
categories have no known truth and are not part of this comparison. The
decision-agreement column is an argmax and therefore inherits the fragility the
stage is about, which is why it is reported beside the continuous statistic
rather than instead of it. The 95th percentile of a material's own contribution
is immune to the tail test only for contamination thinner than 5 percent of the
mass, which is stated where it is reported.

---

## 6. Inputs and outputs

**Read.** The synthetic corpus and the true distributions recovered from it by
replaying the generator; the goodness-of-fit and cross-validated scores the
second notebook writes, which is where the replacement for the retired target
comes from; and the frozen extract of real declarations.

**Written.** One source module and its test file; fifteen new cells and five
edited ones in the third notebook, plus a marker line on eight older figure
cells; fourteen new result tables; **one** figure, two others having been built
and cut inside the stage; twenty-two decisions in the project's decision log,
numbered 143 through 164; twenty-two manuscript discrepancy entries, numbered
131 through 152, with entry 7 marked resolved and entry 95 corrected in place;
the figure style guide; the figure renderer and its test; the mechanics
documentation; and this file.

**Not touched.** The generator, the corpus's values, the extract of real
declarations, the fitting methods, the scoring criterion, the published flip
thresholds, the run against the true distributions, and the manuscript.

---

## 7. Next stage

**Stage 2h**, the sweeps, and its first item is the weight model rather than
anything this stage produced.

**What this stage hands it.** Two constraints, two new sweep items and one
caution.

**The first constraint.** The guard that stops a fitted lognormal running away
into a far tail is the only thing making the level metrics this stage recommends
safe to report. When that guard is swept, the tail correction in the scoring
criterion must stay in force and the fitted-model spread ratio must be reported
at every value tried, because the criterion without that correction cannot see
the failure at all -- it returns the same number to seven significant figures
whether the misplaced mass sits at a hundred times the dataset mean or a
thousand.

**The second constraint.** Adding a family to the comparison is only fair if the
family can use the data. A uniform fitted to n declarations is the smallest and
largest of them and nothing else, so putting it in the main comparison would
manufacture a win for this paper's own method. Uniform and triangular go in the
judgment arm with the pedigree matrix, where the question is how far a model
built without data sits from one built with it.

**Two things to sweep that this stage found rather than inherited.** An upper
truncation of each fitted model, which would remove the thin-far-tail failure
mode outright. And the mode-share concentration, which is already on the list
and which this stage's measurement makes more interesting: every one of the
worst runaway tails in the study is an EQUAL-WEIGHTED fit to a small dataset,
and the market-share-weighted twins do not have the problem.

**The caution.** Every claim of the form "method X is best" in this project is a
claim about one metric, and on two of the seven metrics measured here no method
separates from the runner-up at all. Two methods lead where a leader can be
named; the ordering of the six is negatively correlated with the study's own
headline metric on three of the six companions. A sweep that reports a winner
should name the metric it won on and show the interval.

### Habits, added by this stage

19. **A metric that every method agrees about is not therefore a good metric.**
    The uncertainty index has the lowest disagreement between methods of any
    output in the study and every method is out by about half its own spread.
    Report the agreement and the accuracy together or neither.
20. **Check whether a divisor is a constant that used to be a measurement.** The
    fudge factor here was exactly right under a construction that had already
    been replaced, and the comment beside it had been conceding the problem for
    four stages.
21. **When a stage adds a companion, re-read the paper's existing claims against
    it.** Three of them did not survive, and finding that out was worth more
    than the companion.
22. **Run the notebook before assuming it runs.** The first full run died twenty
    minutes in on a defect an earlier stage had left: a cell iterating a label
    file rather than the frame's own columns, after that stage added a label
    without adding the column.
23. **A leader is not a leader until its interval clears the runner-up.** This
    stage drew a figure naming a best method on all seven metrics; on two of
    them the top two overlapped, and one of the three "different methods" the
    headline counted was that overlap.
24. **Never divide by the best of a set to say how much the set varies.** The
    first scorecard expressed the spread of six errors as a multiple of the
    smallest, and returned 6,508 percent on a row whose best method was almost
    exactly right. The denominator has to be the size of the thing being
    claimed.
25. **Sweep a knob continuously before drawing it.** Three round decades drew as
    three points and hid the only interesting feature: that the criterion goes
    blind exactly at the top of its own grid, and not before.
26. **A symlink is a write path, not just a read path.** A scratch directory of
    symlinks into the results is a good way to let a figure read production
    data and a direct route to overwriting it, because writing a file follows
    the link. Guard the write, not the read.
27. **Check a claim about the manuscript against the manuscript.** "The study
    computes the uncertainty index and reports it nowhere" was carried forward
    from an earlier stage and repeated here; the manuscript reports it in two
    figure panels, defines it in the supplement and draws a conclusion from it.
    The recommendation survived, but it changed from "introduce this" to
    "promote this", which is a different instruction.
