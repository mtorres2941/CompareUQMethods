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

**Stage 2h answers the objection "you only tested one variant" for eighteen
separate choices the study makes.** Each one is a sweep with a tabulated
result, not a spot check. Several changed something the paper says, three
closed questions that had been open for several stages, and the rest confirmed
a setting that was already in place -- which is the outcome a robustness sweep
should usually have, and is worth reporting as such.

**THE FOUR THAT MATTER MOST, if the rest is skimmed.** The summary figure was
comparing two different statistics on one colour scale and five of its sixteen
rows were reporting a cancellation rather than an error (section 1). The two
halves of the study weighted their data by different rules on the exact
dimension the paper is built on, and the fix is now applied, which moves
reported numbers (section 3). A trade-off that blocked any improvement to the
synthetic data across three stages and 36 settings turns out to have been an
artifact of that same weighting mismatch, so the case for leaving the synthetic
data alone no longer holds (section 13). And the judgment-based approach most
practitioners use produces a NARROWER spread than the data it stands for, not a
wider one (section 14).

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

**THE RULE IS NOW APPLIED, AT THE AUTHOR'S INSTRUCTION, AND REPORTED NUMBERS
HAVE MOVED.** An earlier draft of this section ended by saying nothing had been
reweighted and that applying it was the author's decision. The author gave that
decision -- "why haven't you pulled the trigger ... sounds like you already
figured out the setting should be 0.5, so what do we need to discuss?" -- and
the real categories now draw their market shares by the same rule as the
synthetic ones.

**What moved, and the control is why it can be trusted.** All twelve
characteristics measured WITHOUT weights are identical to the last digit,
because the change touches only the weights. All twelve measured WITH weights
move, by a typical amount of 0.066 on how spread a category is, 0.066 on the
effect of weighting itself, 0.135 on its entropy and 0.648 on its skew. All six
goodness-of-fit scores move, which is correct rather than alarming: every model
is scored against the weighted version of its own data, so changing the weights
changes the target.

Across the 147 real categories, the median effect of weighting goes from
**0.105 to 0.148** and the median spread from 0.671 to 0.641.

**And it closes the gap it was built to close.** How fast the effect of
weighting fades as a category grows, and the median effect above a thousand
declarations:

    rule                            real     synthetic   above 1,000
    old: a share per declaration   -0.449     -0.181     0.005 / 0.053
    new, knob at 0                 -0.412     -0.349     0.006 / 0.017
    new, knob at 0.5               -0.161     -0.101     0.044 / 0.085

The tenfold disagreement above a thousand declarations that opened this whole
item is now a factor of 1.9.

**A side effect nobody arranged.** The characteristic an early stage named the
worst in the project, and called structural after four attempts to fix it --
how lognormal the real categories look against how lognormal the synthetic ones
look -- **halves**, from 0.65 to 0.33. Nothing was tuned to achieve that.

**Three frozen comparison tables were re-recorded** so the project's automatic
regression checks compare against the new rule rather than the old one, with a
written record of exactly what moved and the unweighted-column control.

> **So what.** The paper currently shows that market-share weighting stops
> mattering once a category has more than about a thousand declarations. That
> finding was largely an artifact of how the weights were invented for the real
> categories: under a rule that matches how markets actually work, the effect
> above a thousand declarations is nine times larger than the paper reports. A
> practitioner with a well-populated category was being told the market shares
> they cannot obtain do not matter, and that is not what the evidence says.

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

**AND THE CHECK THIS SECTION PROMISED HAS BEEN DONE: the cap does not pull the
two halves of the study apart.** At a cap of twice the largest observation the
real categories improve by 0.99 percent and the synthetic ones by 0.87, a
difference of 0.12 of a percentage point, and the difference never exceeds 0.35
anywhere in the sweep.

**It is NOT neutral between the distribution families, though, and that is the
part worth knowing.** The whole effect falls on the lognormal -- 3.5 and 4.1
percent on the real categories, 2.3 and 3.8 on the synthetic -- while the
kernel estimate and the normal move by less than 0.03 percent, because neither
puts any mass beyond the data for a cap to remove.

**Against the TRUTH that asymmetry almost vanishes**, which is what settles it.
Scored against the known distribution rather than against the data it was
fitted to, a cap of twice the largest observation moves the two lognormals by
+0.57 and -0.85 percent -- opposite signs, both under one percent -- and the
other four by nothing. The head-to-head difference between the two leading
methods, which is what the paper reports, moves in the fourth decimal.

> **So what.** This is now a choice between two measured options rather than
> between a measured one and an unknown. Adopting it would move every number in
> the study, for a failure two existing safeguards already keep out of the
> results, so it is the author's call -- and a reviewer asking whether the cap
> quietly flatters one method can be answered with a number: in sample it helps
> the lognormal and nothing else, and against the truth it helps nobody.

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

**AND THE SAME QUESTION ASKED OF THE ANSWER RATHER THAN OF THE FIT**, because
the author asked for it that way: "ultimately, pLCA results are most important,
so should we measure those?" Over 2,000 simulated buildings scored against the
truth, average error across five outputs:

    smoothing rule        kernel estimate,   kernel estimate,
                          equal weights      market shares
    the older rule             25.9 pct          25.4 pct
    the plain current rule     25.3              25.0
    the current rule as used   25.5              24.8

The four methods that do not use smoothing are identical to the last digit
across all three rules, which is the control saying the measurement is picking
up the smoothing and nothing else. **The older rule is worst on every one of
the five outputs under both weightings**, and the safety guard on the current
rule costs 0.19 of a percentage point under equal weights and BUYS 0.15 under
market shares -- so at the decision level it is free, where at the fit level it
costs a little. **Nothing changes.**

**TWO MORE DISTRIBUTION FAMILIES.** Gamma and Weibull were added to blunt the
objection that only two shapes were tested. Weibull ranks below gamma, which an
earlier stage already established is indistinguishable from the
three-parameter lognormal, so the ordering of the paper's own methods is
untouched. **The family list is not short for want of trying.**

## 12. A regeneration that failed, and the instrument that was actually broken

The author asked for the synthetic data to be regenerated with a wider spread,
to close the one gap that has been open since the data was first made. It was
done, every check passed, and the result was wrong by 99.98 percent.

**The failure was not in the generator and the first two explanations were both
wrong.** The first blamed the generated distributions for having a long thin
upper tail that set their average while never showing up in a sample. Measured,
those distributions had an average of 1.012 against data scaled to average 1.0,
and a ratio of average to midpoint of **1.31** against a threshold of 25 -- they
were sound. The second assumed a partial fix to the sampling code would be
enough; it was not.

**What was broken was the code that draws from a known distribution in order to
check the answer against it.** It laid its lookup points out evenly between the
distribution's lower and upper limits. The widened setting pushed those limits
about a hundred million apart, so the spacing between adjacent points was about
18,000 and the ENTIRE body of the distribution -- everything a sample would ever
contain -- fell between the first two points. The check was drawing from a
staircase rather than from the distribution it was supposed to represent.

It is fixed: the lookup points are now concentrated where the probability is,
with points placed into each tail on a logarithmic spacing.

**And a gate now exists so this cannot recur silently.** Every candidate
setting is checked by comparing the sampler's answer against the distribution's
own, at thirteen probabilities from one in a million to all but one in a
million, at four dataset sizes, under both weightings. The shipped setting and
every bounded candidate are wrong by at most **0.05 percent** on any of them.
The rejected setting is wrong by more than 1 percent on **55 percent** of the
distributions it makes, by a typical 37 percent and a worst of 99.7, all of it
at the one-in-a-million point.

**So the rejected setting is genuinely unusable** -- for a third reason, which
is neither of the two first given: no practical lookup table can represent a
distribution whose limits are a hundred million apart.

> **So what.** The study checks its own answers against a known truth, and for
> one afternoon that check was the broken part while everything it was checking
> was fine. Every number in the paper that comes from it was re-verified. The
> lesson is written into the code: a setting is not allowed near the paper until
> the instrument that measures it has been checked on that setting.

## 13. A trade that has blocked the synthetic data for three stages turns out to have been an artifact

**This is the most consequential thing in the stage and it reopens a question
three separate stages closed.**

The synthetic datasets are less spread out than the real categories, and this
is the one way in which they have never matched. Three stages tried to fix it,
on four different controls, across 36 settings. Every one hit the same wall:
anything that widened the spread ALSO inflated the paper's headline quantity --
how much market-share weighting moves a category -- past what the real
categories show. Widening bought a better match on one thing by wrecking the
number the paper is built on. That was recorded as structural.

**It was not structural. It was a consequence of the two halves being weighted
by different rules**, which is the defect section 3 describes and which is now
fixed. With both halves on one rule, bounded widening settings improve BOTH at
once:

    setting                   overall match   spread   weighting effect
    as shipped                    0.222       0.409       0.340
    moderately wider              0.184       0.254       0.160
    wider still                   0.174       0.205       0.153

Every candidate improves the overall match by 4.6 to 7.7 times the run-to-run
noise, and the mismatch on spread roughly halves.

**The proof that it is the weight rule and not luck.** The same synthetic
datasets were scored against the real categories weighted four different ways,
so the only thing that differs between the columns is how the real categories
were weighted. The change in the weighting mismatch, in units of its own
run-to-run noise:

    real categories weighted by     moderately wider   wider still
    the OLD rule                       +4.4               +6.8     the trade
    the new rule, knob at 0            -1.4               -0.4     gone
    the new rule, knob at 0.25         -0.5               +1.8     gone
    the new rule, knob at 0.5          -5.1               -5.3     reversed

**The trade disappears as soon as the RULE changes, before the knob is turned
at all.** The spread column barely moves across all four, as it must, because
how spread the synthetic data is cannot depend on how the real categories were
weighted.

The reason is arithmetic once stated: the new rule raised the real categories'
median weighting effect from 0.105 to 0.148 while the synthetic data sits at
0.092. The synthetic data now UNDERSTATES that quantity by a third, so widening
moves it toward the real categories instead of past them.

**One characteristic does move the wrong way** -- a measure of how many humps a
distribution has, which carries triple weight in the matching score. It was
checked rather than waved away. Reweighting the synthetic data to match the
real categories on it moves the comparison between the two leading methods by
**0.006 and 0.015**, against **0.026 and 0.043** for spread. So the thing that
gets worse is worth about a quarter of the thing that gets better, in the same
direction.

**Nothing has been regenerated, and that decision is the author's.** The
synthetic data has been frozen since an early stage and every number in the
paper moves when it is remade. What this establishes is that the REASON for
keeping it frozen on this question -- that widening costs the headline quantity
-- no longer holds.

> **So what.** The paper currently carries a stated limitation: the synthetic
> data cannot reach the spread of the most variable real categories, so the
> conclusions are not supported out there. That limitation was believed to be
> unfixable without damaging the main result. It is fixable, the damage was an
> artifact of a bug that has since been fixed, and remaking the data is now a
> cost-benefit decision rather than an impossibility.

## 14. The pedigree matrix, sourced at last, cannot reach the spread of real data

The stage compared the study's data-driven methods against the approach most
practitioners actually use when they have no data: the pedigree matrix, which
turns five judgment scores into a spread. The matrix's own table of factors was
not in this project's reference materials, so the spread was swept RELATIVE to
each category's own -- from half to three times it -- and the sourcing was
recorded as owed.

The author supplied the source. All 3,125 score combinations were enumerated
from it.

    best possible scores            spread factor 1.025
    a middling combination                        1.242
    worst possible scores                         1.587
    a typical real material category              1.871

**A pedigree model is systematically NARROWER than the data it claims to stand
for, and 61.9 percent of real categories are wider than its worst possible
score.** End to end the matrix spans a factor of 1.55; the real categories span
1.01 to 50.9. Of the six spread settings the stage swept, only the narrowest is
reachable on a typical category.

**That is not a defect in the matrix.** It answers a different question: how
uncertain is ONE number for ONE process, not how much do products within a
material category differ from each other, which is what a set of declarations
measures. The paper should say so rather than present the two as rival
estimates of the same thing.

**It strengthens the stage's earlier finding rather than undermining it.** The
comparison already showed that a judgment model's SPREAD barely affects the
design decision -- error of 0.097 to 0.121 across a six-fold range -- while its
CENTRE decides everything. The reachable range is narrower still, so the
conclusion holds with more room to spare.

**One arithmetic trap is recorded because this project walked into it.** Every
factor in the published table contributes to the SQUARE of the spread, so a
model quoted as a spread must halve the exponent; quoting the combined factor
directly would double it. A note taken earlier from that paper also mistook two
single-indicator factors for the total range.

> **So what.** A practitioner using the standard judgment approach on a material
> category is not getting a wider, more cautious answer than one who uses the
> data. They are getting a NARROWER one -- about half the spread of a typical
> real category -- which means the judgment route understates uncertainty on
> precisely the quantity it exists to express. That is a stronger statement than
> the paper currently makes and it is worth making carefully.

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
the first decision-log batch; the generator parameter sweep; the absolute-terms
check on the standardized worsening; the 31-second generation scorecard; the
full-scale notebook 3 run; the certification credit and the backwards margin;
**the weight rule applied to the real categories and the bandwidth measured
downstream**; **the reverted regeneration**; **the parent-sampler fix**; **the
re-frozen comparison tables**; **the parent-level gate and the guard's
corrected rationale**; **the sourced pedigree range**; **decisions 190 to
195**; **the two promised checks**; and this file.

**Three of those commits undo or correct work done earlier in the same stage**,
and they are listed rather than squashed because the project requires a number
change to be bisectable: the regeneration was reverted, the sampler it exposed
was fixed, and the rationale written for a guard added on the wrong diagnosis
was corrected in place.

---

## 2. What was asked

Eighteen sweeps, each closing a "you only tested one variant" objection, run as
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

**Three items were added during the stage by the author.** A certification
credit framed as a decision -- "demonstrate a 10 percent reduction with 75
percent confidence" -- at the 5, 10 and 20 percent tiers. The weight rule
APPLIED rather than only measured. And a regeneration of the synthetic data
with a wider spread, which was attempted, failed, was diagnosed wrongly twice,
and was reverted; what it exposed was a defect in the code that checks answers
against a known truth, and that defect is fixed.

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

**THE SECOND CHANGE THAT MOVED COMMITTED NUMBERS: the weight rule, applied.**
An earlier draft of this section said nothing else had moved and that no
dataset had been reweighted. That is no longer true, by the author's decision,
and section 3 above gives the numbers in full. In summary:

- The twelve characteristics of the real categories measured WITHOUT weights
  are identical to the last digit. This is the control.
- The twelve measured WITH weights all move, by a typical 0.066 on spread,
  0.066 on the weighting effect, 0.135 on entropy and 0.648 on skew.
- All six goodness-of-fit scores for the real categories move, because every
  model is scored against the weighted version of its own data.
- The median weighting effect across the 147 real categories goes from 0.105
  to 0.148, and the median spread from 0.671 to 0.641.
- How lognormal the two halves look now differs by 0.33 rather than 0.65.
- Three frozen comparison tables were re-recorded so the automatic regression
  checks compare against the new rule. A fourth change inside one of them is
  unrelated and benign: it gained two columns settled by an earlier stage, and
  no value it already held moved by more than a millionth of a millionth.

**WHAT WAS CHANGED AND THEN CHANGED BACK, recorded because a reader of the
commit history will see it.** The synthetic data was regenerated with three
widened generator settings and the result was reverted in full, along with all
three settings. The generator's shipped configuration is exactly what it was.
**No paper number survives from that attempt.** What does survive is a fix to
the code that draws from a known distribution in order to check answers against
it, which was broken for wide distributions and is now correct; section 12
gives the detail. That fix changes no committed number, because the setting
that provoked it was reverted and the shipped setting never triggered the bug
-- which was verified rather than assumed, at thirteen probabilities on 240
distributions.

**NOTHING ELSE MOVED.** The crossing constants are unchanged -- the new rule
adds columns and edits none, and the constant the first notebook reads to turn
a weighting risk into a probability is a computational constant rather than
prose. The judgment arm, the upper truncation, the extra families, the
generator parameter sweep and the widening candidates are all measurements:
none is applied in the production path and the synthetic data as shipped is
untouched.

---

## 5. What is still open

### Carried forward, with what this stage did to each

| Item | State |
|---|---|
| **The weight model.** The two halves drew market shares by different rules | **RESOLVED AND APPLIED**, by the author's decision during the stage. The fix is validated against the true hump labels at a setting of 0.5 and its concentration anchors on published production volumes. Reported numbers moved; section 3 and section 4 give them. The tenfold disagreement above a thousand declarations is now a factor of 1.9 |
| **The corpus's joint modality-and-dispersion structure**, coupled to the weight model | **STILL OPEN, and the ground under it has shifted.** The joint gap is mostly the known shortfall in spread rather than a separate defect: the corpus reaches 0.240 multimodal against a real 0.315, which is close, and 0.058 dispersed against a real 0.269, which is a factor of 4.6. **What has changed is that the shortfall in spread is now fixable** -- section 13 shows the trade that made it unfixable was an artifact of the weighting mismatch, which is now repaired. **The hump-spacing fix has still NOT been re-measured against the settled weight rule, and that is now the single most valuable measurement left**, because the reason it was rejected was the same artifact |
| **The profile-likelihood guard on the lognormal** | **RESOLVED.** Confirmed at its current value on the criterion it was chosen by, with the tail correction in force throughout and the largest fitted spread reported at every point. The two other bounds, never previously measured, are an order of magnitude clear of binding |
| **An upper truncation of each fitted model** | **MEASURED. Author decision.** A cap at two to three times the largest observation is not a cost against the truth and removes the failure mode. The promised check is done: it does not pull the two halves apart (0.12 of a percentage point), and its in-sample help to the lognormal alone does not survive scoring against the truth |
| **The pedigree matrix, a uniform and a triangular** | **RESOLVED, and the sourcing gap is CLOSED**: the author supplied the source during the stage and all 3,125 score combinations are enumerated from it. The matrix spans a spread factor of 1.025 to 1.587 against a typical real category at 1.871, so a pedigree model is systematically NARROWER than the data and 61.9 percent of real categories are wider than its worst score. Section 14 |
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
| **THE CASE FOR LEAVING THE SYNTHETIC DATA ALONE NO LONGER HOLDS, and remaking it is now an author decision rather than an impossibility.** Section 13. Bounded widening settings now improve the overall match by 4.6 to 7.7 times the run-to-run noise AND improve the paper's headline quantity, where for three stages they always damaged it. **Anything acted on here must re-run the parent-level gate of section 12 first** |
| **A candidate generator setting must now pass a parent-level gate before its matching score means anything.** The gate exists and every current candidate passes it. The next session extending the generator sweep should run it on whatever it tries; the rejected wide setting is the worked example of a configuration that scores well and is unusable |
| **Which of the two error definitions each published sentence means** is decided nowhere but in this file, and the same is now true of which spread settings in the judgment arm are pedigree models and which are sensitivity |

### Known and accepted

The judgment arm is scored against the market-weighted true distribution, which
exists only on the synthetic half; the real categories have no known truth. The
design comparison in the judgment arm uses 300 pairs against the study's own
2,500, which is enough to separate a factor of two but not to resolve a
difference of a few thousandths. The proxy validation for the weight rule uses
600 synthetic datasets of 10,000. The widening candidates were scored at 440
synthetic datasets per configuration at three random starts each, not at the
full 10,000, so they establish a direction and a rough size rather than a final
number. The pedigree range is computed from the published factor table for a
building material; a different flow type carries a different starting value and
would shift the range slightly, though not by enough to reach a typical real
category.

---

## 6. Inputs and outputs

**Read.** The synthetic corpus and the true distributions recovered from it; the
frozen extract of real declarations and its record metadata; the
goodness-of-fit and cross-validated scores; the run against the true
distributions; and the visible-hump counts.

**Written.** Six source modules or module sections with their tests; eleven
audit scripts and one audit README; several notebook cells changed and three
tables given extra columns; twenty-three decisions in the project's decision
log, numbered 174 through 196; eight manuscript discrepancy entries, numbered
158 through 165; a new section of the mechanics documentation; three re-frozen
comparison tables with a written record of what moved in them; and this file.

**CHANGED IN THE PRODUCTION PATH.** How the real categories draw their
market-share weights, which is the one intended number-moving change of the
stage. And a repair to the code that draws from a known distribution in order
to check answers against it, which moves no committed number because the
setting that exposed it was reverted and the shipped setting never triggered
it.

**Not touched.** The generator's shipped configuration, the synthetic data's
values, the extract of real declarations, the fitting methods used in the
production path, the scoring criterion, the published crossing constants, and
the manuscript.

**Changed and changed back.** Three generator settings and a regenerated
synthetic corpus, reverted in full. The commits are kept rather than squashed
so the sequence can be bisected, which this project requires of anything that
moves a number.

---

## 7. Next stage

**ONE DECISION COMES FIRST AND EVERYTHING ELSE WAITS ON IT: whether to remake
the synthetic data.**

The weight rule is applied, so the coupling the stage was told to respect has
been resolved in the direction that unblocks the corpus. The case for leaving
the synthetic data alone rested on a trade -- widening its spread always
inflated the paper's headline quantity -- and section 13 shows that trade was
an artifact of the weighting mismatch which is now repaired. Bounded widening
settings now improve the overall match by 4.6 to 7.7 times the run-to-run noise
AND improve the headline quantity.

**What a session acting on that must do, in order.** Re-measure the
hump-spacing fix for the corpus's joint modality-and-dispersion structure
against the SETTLED weight rule, because it was rejected on the same artifact
and has never been tested under the repaired comparison; this is the single
most valuable measurement left. Run the parent-level gate on any candidate
before trusting its matching score. Judge the one characteristic that moves the
wrong way -- a measure of how many humps a distribution has, worth about a
quarter of what spread is worth. And expect every number in the paper to move,
which is the cost that has kept the data frozen since an early stage and which
has not become smaller.

**THE SECOND DECISION, independent of the first.** Whether to adopt the upper
truncation, which is free against the truth, removes a failure mode, does not
pull the two halves apart and does not quietly flatter one method.

**THEN Stage 3, the figures.**

**NO SWEEPING IS LEFT UNFINISHED.** The generator-parameter sweep, the
bandwidth sensitivity at the fit level and through the simulation, the widening
candidates, and both promised follow-up checks are all complete and tabulated.

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
33. **Check the instrument before blaming what it measures.** A regeneration
    was rejected because the run that scores answers against a known truth
    reported 99.98 percent errors. The generated distributions were fine; the
    code that draws from them could not resolve a body inside a support a
    hundred million wide. Two diagnoses were published before the third was
    correct, and the first of them had a guard written for it that does not
    fire on the case it was written for.
34. **A verification script that cannot fail is not a verification.** One check
    in this stage printed CONFIRMED unconditionally regardless of what it
    measured, and two checks of the same identity divided by the wrong sign.
    The working version prints the residual and the correlation and lets the
    reader see them.
35. **When a finding has stood across several stages, ask what ELSE changed
    before trusting that it still holds.** The dispersion-versus-weighting
    trade was real in every one of the 36 settings that found it, and it
    evaporated the moment an unrelated defect in how the two halves were
    weighted was repaired. The finding was never wrong on its own evidence; its
    evidence had a shared cause nobody had isolated.
36. **Two ratios of the same two quantities are not the same ratio.** The
    pedigree range was first reported as model spread over data spread against
    a sweep that scales the EXCESS over 1, which moved the reachable band from
    0.028-0.674 to 0.55-0.85 and would have named the wrong swept points as
    attainable.
