# HANDOFF stage-2h - The robustness sweeps

US spelling throughout, as in every file this project writes.

**HOW TO READ THIS FILE. Its reader does NOT have this repository** -- no
project brief, no other report, no table, no figure, no source. Every claim
below is therefore stated in full where it is made, and a trailing `decision N`
or `entry N` is a citation into the project's decision log or its manuscript
discrepancy log, never the substance of the sentence. Where a file has to be
opened, the instruction is addressed to the NEXT CLAUDE CODE SESSION and says
so. The figure is embedded and its caption carries the numbers, so a reader
whose copy does not render the image loses nothing.

---

# IF YOU READ ONE PAGE, READ THIS ONE

**Stage 2h answers the objection "you only tested one variant" for fifteen
separate choices the study makes.** Each one is a sweep with a tabulated
result, not a spot check. Four of them changed something the paper says, three
closed questions that had been open for several stages, and the rest confirmed
a setting that was already in place -- which is the outcome a robustness sweep
should usually have, and is worth reporting as such.

Each claim below carries the number behind it and a plain-language **so what**
for a reader who builds buildings rather than statistical models.

## 1. A correction to the study's own summary figure, and two published sentences reverse

The figure that scores every claim a probabilistic LCA makes against the truth
had **five of its sixteen rows computed the wrong way**. Those five averaged
the error over many buildings BEFORE taking its size, so a method that was too
high on one building and too low on the next reported almost no error at all.
The other eleven rows took the size of the error on each building first. The
figure was drawing two different statistics on one color scale.

All sixteen rows are now the error in a **single** decision. Best method, as a
percentage of how big the true answer is:

    how often a specification cap binds          0.48  ->  30.62
    what a specification cap saves               0.56  ->  38.29
    a cap's chance of saving 5 pct of a building 1.09  ->  32.97
    what using 25 percent less of a material saves 0.00 -> 10.02
    the probability one design beats another     0.81  ->  11.95

**The eleven other rows are unchanged to the last digit**, and the old values
are reproduced exactly by the new column that reports the average-over-many
form, which is the check that this is a change of DEFINITION and not of
arithmetic. Both are kept and both are labelled: the error in one decision is
what a designer choosing between two options carries, and the error in the
average is the right quantity for a portfolio of buildings or a national stock
model.

**TWO SENTENCES IN THE PAPER REVERSE.**

The first is "on what using 25 percent less of a material saves, every method
is exactly right". That is true of the average over many buildings and false
for one: on a single building every method is **10.0 to 13.4 percent** out.

The second is "on how often a specification cap binds, the best method is 0.5
percent out, so picking the right method is nearly the whole problem". Per
building the best method is **30.6 percent** out and the choice of method adds
**23.1**, so most of the error is there whatever you choose.

> **So what.** The headline of the study's summary figure moves from "under the
> best method a probabilistic LCA is right to 0.8 percent on a design
> comparison" to **"right to 12.0 percent"**. That is still the most reliable
> thing it does -- the next best claim is off by 22 percent and the worst by 44
> -- but the earlier number described the average over thousands of
> comparisons, not the one comparison a designer is actually making.

## 2. Half the printed digits on the study's published constants are not measurements

The study publishes six constants of the form "the answer changes 1 percent of
the time once two models are this far apart". Each is read off a curve fitted
through the data, and the uncertainty published beside it is the uncertainty of
THAT curve's parameters -- which says nothing about whether the curve is the
right shape.

Fitting each one a second way, with a method that assumes only that the
probability does not fall as two models separate, **the second answer lands
outside the published uncertainty range on five of the six**. For example, the
lead one material needs over the next before the choice of method stops
changing which leads: the published value is 2.13 with a range of 2.09 to 2.17,
and the second method gives **2.22**.

So the rule now is: print the digits the two methods agree on, plus the first
one they disagree at, and print both values where they still disagree there.

    the safe lead at a 1 pct risk     2.1 against 2.2
    the safe lead at a 5 pct risk     1.64 against 1.61
    the safe lead at a 10 pct risk    1.5 against 1.4

**Nothing is thrown away and no constant moves.** The result tables and the
supplement carry both fits and the full-precision interval, because a reader
checking the work needs the unrounded value. Only prose and figure labels
round.

> **So what.** A reader who refits one of these constants with a different
> standard method gets a different third digit and reasonably concludes
> something is wrong. Printing that digit claims a precision the study does not
> have, and implies a sharp cliff where the real curve is smooth.

## 3. The two halves of the study weighted their data by different rules, and the fix is measured

This is the largest thing the stage was asked to look at. The study compares
methods on two bodies of data: 147 real material categories pulled from a
public database, and 10,000 synthetic datasets generated to look like them.
**They drew their market-share weights by different rules**, on exactly the
dimension the paper is built on.

The real categories drew a share for each individual declaration at random,
independently of its carbon coefficient. The synthetic datasets attached a
share to each hump of the distribution and split it within the hump, so share
was CORRELATED with the coefficient.

That is not a small difference. Weights drawn independently of the values wash
out as a category grows, so the measured effect of weighting MUST fade as
n^-1/2 whatever real markets do -- and real market share does not become more
even as more manufacturers publish. Measured, the median effect of weighting by
category size, and how fast it fades:

    the real categories, as they stand      0.214 0.119 0.068 0.005   fades at -0.449
    the synthetic datasets, as they stand   0.157 0.126 0.060 0.053   fades at -0.181
    the synthetic VALUES, reweighted by
      the real arm's rule                   0.133 0.094 0.038 0.012   fades at -0.367

**The third row is the proof.** Reweighting the synthetic arm's own values by
the real arm's rule reproduces the real arm's behavior, so the gap is the RULE
and not the data.

**THE FIX IS VALIDATED WHERE THE TRUTH IS KNOWN, which is available nowhere
else in this study.** The synthetic datasets have BOTH the true hump labels and
the values, so a rule that guesses the humps can be scored against the rule
that knows them. Cutting the sorted declarations into contiguous groups, with a
knob controlling how tightly the groups track the coefficients:

    knob    effect vs the true labels   how well it orders the categories
    0.00            0.55                        0.850
    0.25            0.65                        0.921
    0.50            1.15                        0.935
    0.75            1.49                        0.877
    1.00            1.60                        0.860

**At a setting of 0.5 the guess reproduces what the true labels give**, to
within 4 percent on a typical dataset, and orders the categories more like the
truth than any other setting. That is a measured anchor rather than a choice.

**AND THE CONCENTRATION ANCHORS INDEPENDENTLY ON PUBLISHED PRODUCTION DATA.**
Drawing the number of market groups the way the generator draws its own humps
-- uniform between one and five, independent of how many declarations there are
-- gives a largest-group share of **0.627**. Published figures put 63.75 percent
of world steel on one production route and 54 percent of global output in one
country. The group count was set by matching the generator; that it reproduces
the published market concentration is a check nobody arranged.

**THE SETTING OF ZERO IS NOT THE NEUTRAL CHOICE AND WAS EXPLICITLY REJECTED.**
It is the claim that market share is unrelated to carbon intensity, and the
published volumes contradict it: the 63.75 percent route is the
HIGHER-carbon one and the low-carbon route is the 0.03 percent one.

**AND THE RESIDUAL DIFFERENCE BETWEEN THE TWO HALVES IS NOT THE WEIGHTS AT
ALL.** Once both use one rule, and each category's effect is divided by how
spread its own values are, the two halves agree in every size band at every
setting of the knob:

    at a setting of 0.5   real       0.396  0.242  0.155  0.158
                          synthetic  0.402  0.258  0.167  0.142

So what is left is a known property -- the effect of weighting scales with how
spread the data are -- with different spread fed into it. **The weight-rule
problem and the synthetic data's known shortfall in spread are the same problem
seen twice**, which is why the stage was told not to attempt them separately.

**NOTHING IN THE PRODUCTION ANALYSIS WAS REWEIGHTED AND NO REPORTED NUMBER HAS
MOVED.** Applying the rule would change every weighted characteristic of the
real arm, which the generator is calibrated against, and that is the author's
decision.

> **So what.** The paper currently shows that market-share weighting stops
> mattering once a category has more than about a thousand declarations. That
> finding is partly an artifact of how the weights were invented for the real
> categories, and under a rule that matches how markets actually work it
> largely disappears. The paper must say so.

## 4. A knob nobody had measured is as noisy as the one everybody quotes

The synthetic data generator is tuned by a single objective, and every previous
stage has judged a proposed change against that objective's **seed-to-seed
noise of 0.0066**. Nobody had measured a second source of movement: every one
of these comparisons redraws the real arm's market shares, and that draw moves
the objective too.

Six independent redraws of the whole arm per rule:

    rule                              objective            central quantity
    as it stands today            0.2355 +/- 0.0148      0.1338 +/- 0.0727
    the ported rule, knob 0.0     0.2372 +/- 0.0064      0.1578 +/- 0.0331
    the ported rule, knob 0.25    0.2453 +/- 0.0115      0.2076 +/- 0.0469
    the ported rule, knob 0.5     0.2490 +/- 0.0060      0.3415 +/- 0.0930

**The weight-draw noise is 0.006 to 0.015, as large as or larger than the
generator seed noise every stage has been quoting.** So the objective's movement
between today's rule and the ported one at knob 0 is 0.0017 against a noise of
0.015 and is not a movement at all.

> **So what.** A later stage that measures a calibration change must quote this
> noise and not the 0.0066, or it will report a change that is a coin toss.

## 5. One weight draw is not the distribution, and a published R-squared depends on which

A per-category weighted number and the arm-wide version of the same number are
not the same quantity, and the paper does not currently distinguish them.
Twenty-five independent redraws of the whole real arm's market shares:

    per category    the central weighting quantity has a typical spread of
                    0.046 on a median of 0.108 -- **47 percent** -- and a worst
                    range of 1.37. One shape statistic ranges by 904.
    arm wide        the median of the same quantity is 0.0981 +/- 0.0052 --
                    **5 percent** -- and the two exponents of the paper's
                    practitioner rule are stable to 5 percent.

A factor of nine between the two. **The control is exact**: every statistic that
does not use the weights moves by precisely zero.

**AND ONE PUBLISHED NUMBER IS A CASUALTY.** The paper reports a rule predicting
how much weighting matters from two numbers a practitioner already has, with an
R-squared of **0.991**. That is computed on the median over a thousand draws. On
a single draw the same rule explains **0.824**, and the correlation with spread
is 0.573 against a published 0.731. **The characteristic stored in the published
data table IS a single draw**, so a reader recomputing the rule from it will not
reproduce 0.991 and should not expect to.

> **So what. A statement about one material category -- "for a category like
> yours, market shares move the answer by 13 percent" -- carries a 47 percent
> uncertainty and has to be given as a range.** A statement about the whole
> collection of categories is nine times steadier and is safe as a number.

## 6. The methods most readers actually use, placed on the same axis

Every method this study compares turns a SET of declarations into a
distribution. The methods in general use do not: the pedigree matrix, which is
what the main life-cycle databases apply, is expert judgment applied where data
is absent, and a uniform or triangular range is what someone reaches for with
two or three numbers and no dataset.

They cannot be compared like for like and this stage does not claim they can.
What makes the comparison possible is the yardstick: the error against the true
distribution the data came from does not care how a model was built.

**Two dimensions were swept, not one, and the second turned out to be the one
that matters.** A pedigree model is a SPREAD applied around a POINT ESTIMATE
the practitioner already holds, and nothing puts that point where the
category's true average is.

**AT THE FIT LEVEL a judgment-driven model is 2 to 100 times further from the
truth** than the best data-driven fit. The realistic case -- the point estimate
taken from one declaration the practitioner happened to obtain -- is 4 to 12
times worse.

**AT THE DECISION LEVEL a well-centered one is competitive.** On the question
"is option B better than option A", over 300 design pairs, the six data-driven
methods are out by 0.080 to 0.114; a pedigree model centered on the category's
true average is out by **0.097 at a matched spread, and 0.097 to 0.121 across a
six-fold range of spread**. The spread hardly matters.

**WHAT BREAKS IT IS THE CENTER.** And how the center is wrong matters more than
how much:

    every material displaced by the same fraction   no effect at all, exactly
    each material displaced independently, 10 pct   0.111
    each material displaced independently, 25 pct   0.159
    each material displaced independently, 50 pct   0.223
    the realistic case, one declaration each        0.170 to 0.271

A displacement applied to every material alike cancels exactly, because both
designs' totals scale by the same factor. **Sweeping only that would have
reported a null that was an artifact of the sweep.**

**And the shape matters too**: at a matched spread and a correct center, the
pedigree lognormal is out by 0.097 where a uniform is out by 0.206 and a
triangular by 0.198.

> **So what.** The deliverable sentence is: a judgment-driven model with a
> plausible spread gives the same design answer as a data-driven one PROVIDED
> its point estimate is not displaced -- and what a practitioner actually has,
> one declaration per material, displaces it independently for each one, which
> roughly doubles the error. **The spread is where the pedigree matrix puts all
> its effort and the center is what decides the answer.**

**ONE SOURCING GAP THE MANUSCRIPT MUST CLOSE.** The pedigree matrix's own table
of uncertainty factors is not among this project's reference materials, so the
spread was swept RELATIVE to the data's own rather than in absolute pedigree
units. That answers the question without asserting a table from memory -- which
this project has had to withdraw once before -- and it means **a specific
pedigree score cannot be placed on this axis until that table is obtained.**

## 7. Three questions closed, and one setting confirmed

**The industry-average declaration check is dropped, on the data.** The idea was
to compare a category's plain average against an industry-average declaration,
which is in principle the production-weighted average the study says nobody has.
The database does flag declaration types and the frozen extract does record
them: **all 120,280 records read "Product EPD"**. There is no industry-average
declaration to compare against. Guessing which declarations are industry
averages from their product names was refused.

**There are no six-mode datasets.** The study has carried a note since an early
stage that 5.6 percent of the synthetic datasets have six or more humps against
0.7 percent of real categories. Measured at the smoothing the study actually
uses, the synthetic datasets run 76 / 21 / 2.4 / 0.4 / 0.1 percent over one to
five humps with a **maximum of five**, and the real categories 68 / 26 / 5 with
a maximum of three. The inherited figure counts humps a different way on a
version of the data that has been regenerated twice since.

**Splitting the categories into products did not manufacture the result.** An
earlier stage resolved the database's material categories into specifiable
products. Rerunning every headline against the ORIGINAL unsplit categories:

    method                      split   unsplit   deduplicated
    lognormal, equal weights    0.362    0.358       0.386
    kernel estimate, equal      0.260    0.236       0.244
    normal, equal weights       0.118    0.098       0.134

The ordering is identical and the shares move by at most 0.03. The unsplit
categories are harder to fit, as expected, because they mix products.

**And the guard on the lognormal fit stays where it is**, confirmed on the
criterion it was chosen by, with the two other bounds nobody had ever measured
also checked: both are an order of magnitude clear of binding.

> **So what.** Three of the study's open worries turned out to be about things
> that are not there, and one setting was confirmed rather than changed. That
> is what a robustness sweep is supposed to produce most of the time.

## 8. Capping the top of each fitted model is free, and the decision is the author's

A goodness-of-fit score charges a model for how much mass it misplaces and not
for how FAR out it puts it, so a model can score well and still wreck the
simulation that samples from it. Capping every fitted model at a multiple of
the largest value actually observed would remove that failure outright, at the
cost of one assumption.

Measured against the true distributions, on 400 datasets:

    cap at ... times the largest      error against truth   worst model spread
      1 (the largest value itself)          0.1957                1.45
      2                                     0.1917                1.75
      3                                     0.1916                1.90
     10                                     0.1918                2.54
      no cap                                0.1918                5.40

**A cap at two to three times the largest observation cuts the worst runaway
fit from 5.4 times the data's own spread to 1.8, and is not worse against the
truth -- it is better in the fourth decimal.** Capping at the largest observed
value itself is clearly harmful.

> **So what.** This is now a choice between two measured options rather than
> between a measured one and an unknown. Adopting it would move every number in
> the study, for a failure two existing safeguards already keep out of the
> results, so it is the author's call.

## 9. Every shape parameter of the generator is already at its best value

Six parameters swept, sixteen configurations, using the project's existing
tuning machinery rather than a second copy of it. The objective measures how
closely the synthetic datasets resemble the real categories across every
characteristic at once, and it has a known run-to-run noise of 0.0066.

    configuration            objective   away from the default,
                                         in units of that noise
    as it ships                0.2251           --
    min_q1_over_iqr 0.05       0.2301          +0.8
    min_q1_over_iqr 1.0        0.2446          +3.0
    min_mode_sd_frac 0.05      0.2251           0.0  (identical)
    min_mode_sd_frac 0.25      0.2404          +2.3
    trunc_iqr_mult 2           0.2516          +4.0
    trunc_iqr_mult 8           0.2809          +8.5
    mode_coupling 0.0          0.2123          -1.9
    mode_coupling 0.5          0.2057          -2.9
    mode_share_alpha 1         0.2582          +5.0
    mode_share_alpha 3         0.2275          +0.4
    point_weight_alpha 0.3     0.2890          +9.7
    point_weight_alpha 3.0     0.2468          +3.3

**Every one of the four shape parameters is best where it already sits.**

**One inherited expectation is reproduced and does not survive.** An earlier
stage measured the truncation floor at 0.05 and found the spread of the
synthetic data matching the real data much better. It does -- that distance
improves from 0.380 to 0.207, which is the same finding -- and the overall
objective gets worse, because the gap between the two halves of the study on
the paper's own central quantity, how much weighting matters, **more than
doubles**. Measured undivided, so that this is not an artifact of how the
comparison is scaled: the spread gap closes from 0.2685 to 0.1463 while the
weighting gap opens from 0.0285 to 0.0583, and the scaled and unscaled
versions agree in direction on all ten characteristics. **It buys spread and
pays for it with the thing the paper is built on.** That is the same trade this
project has now recorded on four separate knobs.

**One parameter turns out not to be a parameter.** The minimum hump width
returns an objective identical to the last digit at 0.05 and 0.10, because the
rejection it controls never fires below about 0.15: zero times in 300 draws at
0.05, once at the shipped 0.15, ten times at 0.25 and 103 at 0.5. The shipped
value sits exactly where it begins to bite.

**THE TWO CONFIGURATIONS THAT BEAT THE DEFAULT ARE THE OBJECTIVE ASKING FOR THE
WRONG FIX, and this is the most important row in the table.** Turning the
synthetic arm's weight coupling DOWN improves the objective. Every
characteristic that does not involve the weights is untouched to the last
digit, so this is purely the weights -- and what it does is make the synthetic
arm draw its weights the same way the real arm currently does, which is the
assumption published production volumes contradict. It would also destroy the
true distribution every run against the truth is scored against.

> **So what.** The objective measures how well the two halves of the study
> AGREE, and the cheapest way to agree is for both of them to make the same
> false assumption. A later stage looking at that table will see two
> improvements and must not take them. This is why the weight rule in section 3
> is ported in the other direction.

**And the author's own proposal is still blocked rather than answered.** Moving
the hump-share concentration from 10 to 1 measures 5.0 noise units worse, which
reproduces an earlier stage's number exactly. That earlier stage said the
comparison had to be redone once the two halves weighted their data the same
way. It is redone here under the same mismatched rules, because they still do
not -- so the question is waiting on the author's decision about the weight
rule, not on a measurement.

## 10. The certification credit: near the bar, the methods disagree two thirds of the time

**THE QUESTION, in a practitioner's words.** Green-building certification
awards points for demonstrating a reduction against a baseline -- typically 10
percent. Under a probabilistic LCA that claim naturally becomes "demonstrate a
10 percent reduction with 75 percent confidence". Does the choice of UQ method
change whether you earn it?

The study already computed the probability and scored its error. It had never
asked the DECISION that probability is used for, which is a different and more
fragile question: it inherits the error in the probability AND a cliff at the
threshold.

**THE ANSWER**, on 600 design pairs built at five different true savings, 3,000
cases in all:

    the truth earns the credit                      17.4 pct of cases
    at least two of the six methods disagree        18.2 pct
    the best method calls it wrong                   8.8 pct
    the worst method calls it wrong                 12.5 pct

**AND THE FRAGILITY IS THE THRESHOLD, NOT THE METHODS.** Split by how far the
TRUE confidence sits from the 75 percent line:

    distance from the line    designs   the six disagree
    within 0.05                   305       65.3 pct
    0.05 to 0.10                  281       46.3 pct
    0.10 to 0.25                  800       20.5 pct
    beyond 0.25                 1,614        3.4 pct

The same story read by how much the design actually beats the baseline: the
methods disagree on 0.2 percent of designs with no real saving, 6.5 percent at
a true 10 percent saving, 31.3 at 15 percent and 52.0 at 20.

> **So what. A design comfortably over or under the bar is called the same way
> by every method, and a design sitting on the bar is a coin toss.** That is
> not an argument against writing credits this way -- it is an argument for
> writing them with the margin stated, because a certification scheme that
> awards a point at exactly 75 percent confidence is awarding it on the
> modeling choice rather than on the building whenever an applicant is close.
> A normal distribution is the worst method for this on 8 of the 15 tier and
> confidence combinations tested.

**TWO CONSTRUCTIONS WERE WRONG BEFORE ONE WAS RIGHT**, and both are recorded
because the second is a defect in the study's own code.

A credit is a WHOLE-DESIGN claim, not a one-material one. Capping a single
material almost never moves a whole building by 10 percent, so asking the
question of a single-material intervention answers it where nobody is near the
bar: the truth clears it in **0.24 percent** of cases.

And the study's existing comparison margin points the wrong way for this
question. It computes the share of simulations in which the proposal comes in
below the baseline times a margin, and the study's margins are 1.0, 1.05 and
1.2 -- all ABOVE one, which asks "is the proposal better, **or worse by less
than** 20 percent". That is a tolerance. A credit needs the margin BELOW one.
Confirmed on the study's own output: at a true 20 percent saving the 1.2 margin
reads **0.9993** against a plain "is it better" of 0.9628 -- looser, not
stricter. **The code comment described it as "the share in which A beats B by a
margin worth acting on", which describes the other direction.** Corrected, with
a test. **No published number moves**: the 1.05 and 1.2 figures are what the
comparative-LCA literature reports and are correct as tolerances; the sentence
describing them was wrong.

**A SOURCING CONSTRAINT.** The tiers used are the study's own 5, 10 and 20
percent, which match the tiered structure certification schemes use. **The
exact wording, tier and confidence of any specific credit must be sourced
before the paper cites one** -- the same discipline this stage applies to the
pedigree matrix's factor table, and for the same reason.

---

## 11. Two sensitivities that changed nothing, reported because they could have

**THE BANDWIDTH.** The rule for how much to smooth a kernel estimate was
settled two stages ago on held-out likelihood. Swept again: on the synthetic
half of the study the older rule beats the current one on that criterion **61
to 63 percent** of the time, which is the opposite of what the real categories
said. On the study's own criterion they do not disagree -- the current rule
gives 0.1517 against 0.1649 under equal weights and **0.0621 against 0.1016**
under market-share weights, a 64 percent difference.

**What the headline does under the older rule, which is the configuration the
manuscript was written against:** the kernel estimate looks WORSE, by 9 percent
under equal weights and 64 under market-share weights. So the manuscript's
configuration understates this paper's own method, which is the conservative
direction and should be stated rather than quietly corrected.

**TWO MORE DISTRIBUTION FAMILIES.** Gamma and Weibull were added to blunt the
objection that only two shapes were tested. Weibull ranks below gamma, which an
earlier stage already established is indistinguishable from the
three-parameter lognormal, so the ordering of the paper's own methods is
untouched. **The family list is not short for want of trying.**

---

## The figure

![Every claim a probabilistic LCA makes, scored for all six methods against the truth, with the five corrected rows](../outputs/figures/CompareUQMethods_FIG_ClaimScorecard.png)

**Figure: under the best of the six methods a probabilistic LCA is right to
12.0 percent on the design comparison and wrong by 32.0 percent on which
material leads -- and the count of black boxes in the upper panel is not a
ranking of methods, because the ordering inverts with dataset size.**

**EVERY CELL OF BOTH PANELS IS THE SAME QUANTITY**: that method's mean absolute
error against the truth PER DECISION, as a percentage of the mean true level of
the thing being claimed. A black box marks the method closest to the truth in
each row. "Sampled" means the market shares were drawn at random because nobody
publishes them; "equal" means every declaration counts the same.

**WHAT THE CORRECTION DID TO THIS FIGURE.** The four reduction-strategy rows
and the design comparison moved, and nothing else did. Before and after, best
method: how often a cap binds **0.5 to 30.6**, a cap's mean saving **0.6 to
38.5**, a cap's chance of saving 5 percent **1.1 to 33.0**, what using 25
percent less saves **0.0 to 10.0**, the probability B beats A **0.8 to 12.0**.
The eleven other rows are bit-identical and the old values are reproduced
exactly by the new portfolio column.

**READING THE UPPER PANEL.** The building total is recovered to 8.0 percent on
its mean and 22.1 on its standard deviation. Attribution runs from 10.0 percent
on a material's share of the total to **32.0 on its chance of being the largest
contributor**, which remains the worst-recovered claim in that block. The
uncertainty index is **43.6 to 45.5 percent** with only 1.9 points between best
and worst -- the one claim where the choice of method does not matter and no
method is close. The reduction strategies are now the widest block on the
figure: a cap's chance of saving 5 percent runs **33.0 to 58.1**, and the two
normal fits are the whole of that spread. The design comparison is the best
recovered claim at **12.0 to 17.1**.

**THE RIGHT-HAND BAR IS A DIFFERENT QUESTION FROM THE CELLS.** It is worst
minus best, so it is what the CHOICE of method costs, where the cells say how
good the answer is at all. On a cap's chance of saving 5 percent the choice
costs 25.1 and the best method is still 33.0 out. On the uncertainty index the
choice costs 1.9 and every method is about 44 out.

**THE LOWER PANEL IS THE UPPER ONE'S OWN CAVEAT.** The seven per-material
claims pooled, split by the material's own dataset size. The equal-weighted
lognormal is closest below 100 declarations, the sampled-share lognormal from
100 to 999, and the sampled-share kernel estimate above 1,000 at **11.0 against
the equal-weighted lognormal's 17.2**. Both axes turn over: equal weights win
every band below 100 and sampled shares win every band above.

**AND ONE OF THE SIXTEEN ROWS IS REDUNDANT, which the correction exposed.**
"What using 25 percent less saves" and "a material's share of the total" now
carry identical numbers in all six cells, and that is an identity rather than a
coincidence: using 25 percent less of a material removes exactly a quarter of
that material's share of the building, so the error in the first is exactly
0.25 times the error in the second and their relative errors are equal to
machine precision. Verified over all 60,000 rows, correlation 1.00000000.
**Under the old averaged definition both rows read 0.00 and the identity was
invisible.** The paper should report one of them, not both.

---

## 1. Stage and branch

| | |
|---|---|
| **Stage** | 2h, the robustness sweeps |
| **Branch** | `stage-2h-robustness` |
| **Branched from** | `3ed41ae` on branch `stage-2g-metric` |

The commit branched from was made by this stage: the previous branch had
uncommitted work, which was notebook 1's stored execution output. It is
committed there for the record and then CLEARED here, because this repository
requires notebooks to carry no stored output and a test enforces it.

Commits, in order: the scorecard's per-unit correction with its module and
tests; the both-ways rule for fitted crossings; the unified weight rule and the
judgment arm; the correction to the block count and the upper-truncation sweep;
Weibull, the population sensitivities and the arm-gap measurement; the
six-mode and joint-cell tests; the judgment arm at scale with the published
anchor; the weight realizations and the profile bounds; the fix for a crossing
that does not exist, found by a smoke run; the manuscript discrepancy entries;
the decision log; and this file.

---

## 2. What was asked

Fifteen sweeps, each closing a "you only tested one variant" objection, run as
sweeps with tabulated results rather than as spot checks. Before them, one
correction: put all sixteen rows of the study's summary figure on the same
quantity, because five of them averaged the error over many buildings before
taking its size. A standing rule for every fitted constant: fit it both ways,
report both with the interval at full precision, and round only in prose, at
the first digit where the two fits disagree. Then the weight model, which is
the largest item; the corpus's joint modality-and-dispersion structure
alongside it, because the two are coupled; the pedigree matrix as a
two-dimensional sweep over spread AND location; and the remaining sweeps over
the bandwidth, the lognormal's bounds, an upper truncation, two extra
distribution families, the generator's own parameters, the weight
concentration and coherence, multiple weight realizations, and two alternative
definitions of the empirical population.

---

## 3. What was done

**Four new source modules or module sections, all with tests.** One market-share
rule for both halves of the study, with a coherence knob and a block count
matching the generator's own; a judgment-driven arm holding the pedigree
matrix, a uniform and a triangular; an optional upper truncation that wraps any
fitted model including the kernel estimate; and a Weibull family for the main
comparison. Plus the corrected scorecard assembly, which had lived entirely in
a notebook cell, and the both-ways rule for crossings.

**Seven new audit scripts**, each named for what it measures, writing to the
audit tables directory. The scripts are the record rather than the tables, by
an earlier decision of the author's: a script that rewrites its table on demand
is a better record than a large file.

**Two notebook cells changed and three summary tables gained a column.** The
scorecard cell now calls the tested function; the three crossing tables gain
five columns saying what prose may print; and the two summary tables that fed
the five wrong scorecard rows now carry both error definitions.

**ONE DEFECT FOUND BY THE PROJECT'S OWN SMOKE-RUN RULE.** The new rounding rule
computed a decimal count from the logarithm of a value, which overflows when a
crossing does not exist. Under the notebook's smoke configuration -- 20 design
groups rather than 2,500 -- several crossings are unreachable and come back as
infinite. The rule now passes a non-finite crossing through as text. The defect
would otherwise have surfaced forty minutes into the full run.

**ONE DEFECT IN MY OWN FIRST DESIGN, CORRECTED BEFORE IT REACHED A RESULT.** The
first version of the unified weight rule grew the number of market groups with
the size of the category, up to twelve. The generator draws its own component
count uniformly between one and five, independent of size, so a size-growing
count is a different weight model rather than the one being ported -- and it
reintroduces exactly the artifact the port exists to remove, because more
groups at large size means more dilution at large size. Measured and replaced.

**The test suite is 560 tests.** The eight regression fixtures that pin the
dataset characteristics and all six goodness-of-fit scores pass unchanged,
which is the check that the fitting, the corpus and the empirical extract were
not touched. The optional upper truncation defaults to no truncation and is
verified bit-identical by those fixtures.

---

## 4. Numbers that moved

**One change moved committed numbers, and it is a change of definition.**

Five of the sixteen rows of the summary figure, listed in section 1 above. The
eleven other rows are bit-identical and the old values are reproduced exactly
by the new average-over-many column, which is what establishes that the
underlying run is unchanged.

**Consequences inside that figure.** The six methods now differ measurably on
16 of 16 claims rather than 15. The best method changes on four of the five
corrected rows. The most expensive question is still the reduction strategy;
the specific claim inside it moves from "how often a cap binds" at 31.0 percent
to "a cap's chance of saving 5 percent" at 25.1. The figure's own headline
moves from 0.8 percent to 12.0.

**Nothing else moved.** The crossing constants are unchanged -- the new rule
adds columns and edits none, and the constant the first notebook reads to turn
a weighting risk into a probability is a computational constant rather than
prose. The weight rule, the judgment arm, the upper truncation and the extra
families are all measurements: none is applied in the production path, no
dataset was reweighted, the generator was not rerun and the synthetic data was
not regenerated.

---

## 5. What is still open

### Carried forward, with what this stage did to each

| Item | State |
|---|---|
| **The weight model.** The two halves drew market shares by different rules | **MEASURED AND NOT APPLIED.** The defect is confirmed, the fix is built and validated against the true labels at a setting of 0.5, and the concentration anchors on published production volumes. Applying it would move every weighted characteristic of the real arm, which the generator is calibrated against. **Author decision.** |
| **The corpus's joint modality-and-dispersion structure**, coupled to the weight model | **STILL OPEN, and narrowed.** On one definition for both halves the corpus reaches 0.240 multimodal against a real 0.315, which is close, and 0.058 dispersed against a real 0.269, which is a factor of 4.6. **So the joint gap is mostly the known shortfall in spread rather than a separate defect.** Excising the 97 both-at-once datasets moves the headline by a quarter of a percentage point; the cell itself reads the other way, so it does not hide a result. The hump-spacing fix was NOT re-measured against the settled weight model, because the weight rule was not settled into the production path -- that re-measurement is the first thing the next stage should do if the author adopts the rule |
| **The profile-likelihood guard on the lognormal** | **RESOLVED.** Confirmed at its current value on the criterion it was chosen by, with the tail correction in force throughout and the largest fitted spread reported at every point. The two other bounds, never previously measured, are an order of magnitude clear of binding |
| **An upper truncation of each fitted model** | **MEASURED. Author decision.** A cap at two to three times the largest observation is not a cost against the truth and removes the failure mode |
| **The pedigree matrix, a uniform and a triangular** | **RESOLVED as a measurement**, with a sourcing gap named: the matrix's factor table is not among this project's references and a specific pedigree score cannot be placed on the axis until it is obtained |
| **Multiple weight realizations** | **RESOLVED.** A per-category number carries 47 percent relative spread and the arm-wide version carries 5 |
| **The deduplicated and unsplit empirical variants** | **RESOLVED.** The method ordering is identical on all three |
| **An industry-average declaration as a direct estimate of the market-weighted mean** | **CLOSED, on the data.** The extract contains none |
| **Six-or-more-hump datasets** | **CLOSED.** There are none on either half |
| Every figure brought to the style guide; the figure manifest; the older figures still carry a non-ASCII minus sign | Stage 3 |
| A real-building anchor, if citing the staircase paper is not enough | Stage 2i, optional |
| **The corpus's characteristic list omits the modality measure the paper should report** | Still open. Owned by whichever stage next reruns the second notebook |
| **The tension between accuracy and tail-robustness** | Still open and now half-settled: the upper truncation measured here would remove the tail half of it at no cost against the truth |
| **British spellings in files earlier stages wrote** | Still open, owned by the deposit tidy-up |

### Opened here

| Item | |
|---|---|
| **THE GENERATOR-PARAMETER SWEEP IS DONE and is section 9 above.** All sixteen configurations completed. The next session that wants to repeat or extend it should read `audits/GENERATOR_SWEEP.md`, which carries the commands, what each parameter is and how to read a result against the objective's noise -- **including the warning that the two configurations which beat the default are the objective asking for the wrong fix** |
| **THE BANDWIDTH SWEEP IS DONE and is section 11 above.** The shipped rule stands, and what the headline does under the older rule is reported: the kernel estimate looks 9 to 64 percent worse, which is the conservative direction |
| **A per-unit and a per-portfolio error are different questions and the paper now has both for sixteen claims.** Which one each published sentence means is not yet decided anywhere but in this file |
| **The weight-draw noise in the calibration objective had never been measured and is as large as the generator's seed noise.** Any later stage judging a calibration change must quote it |

### Known and accepted

The judgment arm is scored against the market-weighted true distribution, which
exists only on the synthetic half; the real categories have no known truth. The
design comparison in the judgment arm uses 300 pairs against the study's own
2,500, which is enough to separate a factor of two but not to resolve a
difference of a few thousandths. The proxy validation for the weight rule uses
600 synthetic datasets of 10,000.

---

## 6. Inputs and outputs

**Read.** The synthetic corpus and the true distributions recovered from it; the
frozen extract of real declarations and its record metadata; the
goodness-of-fit and cross-validated scores; the run against the true
distributions; and the visible-hump counts.

**Written.** Four source modules or module sections with their tests; seven
audit scripts and one audit README; two notebook cells changed and three tables
given extra columns; sixteen decisions in the project's decision log, numbered
174 through 189; eight manuscript discrepancy entries, numbered 158 through
165; a new section of the mechanics documentation; and this file.

**Not touched.** The generator, the corpus's values, the extract of real
declarations, the fitting methods used in the production path, the scoring
criterion, the published crossing constants, and the manuscript.

---

## 7. Next stage

**Stage 3, the figures**, unless the author takes one of the two decisions this
stage hands back first.

**THE TWO DECISIONS.** Whether to apply the unified weight rule, which would
move every weighted characteristic of the real arm and require the generator's
calibration to be re-measured -- and, if so, whether the hump-spacing fix for
the corpus's joint structure becomes viable under it, which is the coupling the
stage was told to respect and which could not be tested without the rule being
settled. And whether to adopt the upper truncation, which is free against the
truth and removes a failure mode.

**THE TWO PIECES OF UNFINISHED SWEEPING.** The generator-parameter sweep was
running when this stage closed and its commands and reading instructions are
written down; the bandwidth sensitivity was not run and its scripts are in
place.

**THE CAUTION THIS STAGE ADDS.** Two of the numbers it corrected were not wrong
arithmetic but the right arithmetic answering a question nobody had stated --
an error averaged over many buildings where the sentence was about one. Before
quoting any summary statistic in the manuscript, state whether it is about one
decision or about the average of many, because for five of the study's sixteen
headline claims those differ by a factor of twenty.

### Habits, added by this stage

28. **Check whether a summary averages before or after taking the absolute
    value.** Five of sixteen rows on the study's own summary figure did it in
    the order that reports a cancellation as an accuracy, and they were the
    five that flattered the study most.
29. **When porting a rule between two halves of a study, port its PARAMETERS
    too.** The first version of the weight rule ported the mechanism and
    invented its own block count, which reintroduced the artifact the port
    existed to remove.
30. **A sweep over one dimension can report a null that the sweep created.** A
    displacement applied to every material alike cancels exactly in a design
    comparison; sweeping only that would have concluded the location of a
    judgment model does not matter, when it is the only thing that does.
31. **Measure the noise of the thing you are measuring against.** Every stage
    for six stages has judged calibration changes against the generator's
    seed-to-seed noise; the weight draw moves the same objective by as much
    again, and nobody had looked.
32. **A concern inherited as a number may be about a version of the data that
    no longer exists.** "5.6 percent of the corpus has six or more humps" was
    measured a different way on a corpus regenerated twice since; the current
    maximum is five.
