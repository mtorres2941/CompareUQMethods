# Review of Stage 2j: the questions the report has to answer

A fresh window, given `reports/STAGE_REPORT_2j.md` and `reports/STAGE_PROMPTS.md`
and nothing else. Questions were formed from the report first; every one marked
**CONFIRMED** was then checked against the repository and the check is named so
it can be re-run. Ordered by how much the answer changes.

---

## A. Three statements in the report contradict their own tables

**A1. The floor is crossed by the report's own numbers, and the sentence
asserting it cannot be is in bold. CONFIRMED.**

Section 2: "it flattens at 0.084 against a floor of 0.099 ... **No quantity of
EPDs gets a uniform-weighted fit below that floor.**" In
`outputs/tables/audits/TABLE_BandwidthNeff.csv`, `parent_separation` at 1000+ is
**0.0995** and `uniform_best` is **0.0844**. The uniform-weighted fit is already
15 percent below the floor, and has been since 100 declarations (0.0968 against
0.0994). Either `parent_separation` is not a lower bound on W1(uniform fit,
market parent) -- and it is not, because a fit to a finite sample need not sit at
the uniform-weighted population -- or the two numbers are not comparable. Which?
And if it is not a bound, what survives of the bias-variance-with-a-floor story?

Related: the same table's floor row is stated as "~0.10 to 0.12, **at every
size**". Measured it is 0.1226, 0.1132, 0.1131, 0.0994, 0.0995 -- it falls
monotonically by 19 percent. Why does the distance between two *population*
objects depend on the sample size at all, and does that not say the size bands
differ in something besides n?

    ~/miniforge3/envs/compareuq/bin/python -c "import pandas as pd; d=pd.read_csv('outputs/tables/audits/TABLE_BandwidthNeff.csv'); print(d.groupby('band')[['parent_separation','uniform_best','market_best']].mean())"

**A2. "No kernel, no bandwidth, a reader can check it by hand" describes a
quantity computed from a fitted kernel at a swept bandwidth. CONFIRMED.**

Headline 4 and section 2: "no distribution, no kernel, no bandwidth in it ... it
is in the simplest statistic there is, and a reader can check it by hand." The
same section's table caption says "with the kernel estimate at each fit's own
best bandwidth". `audits/weighting_location_shape.py` defines
`LOCATION = |mean(fitted model) - mean(parent)|`, so it is the mean of a fitted,
truncated model, swept over 17 bandwidth multiples. The script's own docstring
makes the correct and weaker claim: "two estimators, **so the bandwidth cannot
be the answer either way**". Why does the report upgrade that to "there is no
bandwidth in it"? A reader cannot check a best-bandwidth kernel fit by hand.

**A3. "The standard error of one" contradicts "the effective sample size is 2.8"
two clauses later**, and it reinstates wording decision 219 withdrew. Decision
219 says in terms: "the earlier wording in the stage report -- 'leaves the
estimate resting on two observations' -- was wrong and is withdrawn." The report
now says "you hold **two observations** of the thing that carries the weight"
and decision 222 repeats it in bold. If this is a refinement rather than a
reversal, say which, because CLAUDE.md forbids reversing a decision silently and
a reader sees the withdrawn sentence back.

---

## B. The claim that the crossover lands at the cutoff rests on 40 to 52 datasets and no interval

**B1. Every test the report runs puts the 81-99 band on a coin flip.
CONFIRMED.** The band holds **52 of 2,500** datasets in the bandwidth audit and
**40 of 2,000** in the location audit -- it is the thinnest band in the design by
a factor of twelve, because the corpus stratifies at 3-9 / 10-99 / 100-999 /
1000-9999 and 81-99 is a sliver of one stratum. Binomial standard errors on the
three rows the report prints as "settled three ways":

    production rule   53.85 +/- 6.91
    plain count       50.00 +/- 6.93
    own best bandwidth 51.92 +/- 6.93

Not one of them is distinguishable from 50 percent. "It crosses half at the
cutoff, for both families, with nothing tuned to put it there" is the load-
bearing sentence of headline 4, and its cell is a coin flip with no count and no
interval anywhere in the report. What is the interval, and what does the claim
become once it is printed?

**B2.** The bands break at 80/81 -- the number being validated. On bands cut
anywhere else, where does the crossing sit? As printed, "crosses half at the
cutoff" is partly a consequence of where the band edge was put.

**B3. The two "independent 2,000-dataset draws" disagree on the SIGN in that
band. CONFIRMED.** Seed 1: uniform 0.0742, known 0.0874 -- known is worse. Seed
2 (the run on disk): uniform 0.0695, known 0.0611 -- known is better. At 10-80
they differ by a factor of 30 (+0.0012 against +0.0351). The paired difference on
the committed run is **-0.0083 +/- 0.0381** at 81-99, which swallows both seeds
and both signs. The report presents the pair as replication -- "gives the same
shape on both seeds". On what reading do they agree at the two bands where the
crossing lives?

**B4. A third run gave the opposite sign in every band and the report does not
mention it.** Decision 222 records it: "A 300-DATASET RUN OF THIS AUDIT SHOWED
THE OPPOSITE SIGN ON LOCATION ... It put the market-weighted location below the
uniform one in every band." Its stated grounds for discarding it -- "two
independent 2,000-dataset draws agree with each other" -- is false at 81-99 and
at 10-80 per B3. Why is this not in the report, given the report is the document
a reviewer reads?

**B5.** The corpus has 10,000 datasets. Why subsample 2,000 for the stage's most
contested result? The full corpus removes the seed question and puts about 200
datasets in the 81-99 band instead of 40. What does the full run give?

**B6.** The reproduce command `python audits/weighting_location_shape.py --n
2000` has no `--seed`, and `--seed` defaults to 0. It reproduces **one** of the
two rows the claim rests on. Which seed produced the other?

---

## C. Four things the prompt required that the report measured and did not report

**C1. The FIT-level result. CONFIRMED to exist and be omitted.** Prompt
deliverable 4: "The same question asked of the FIT as well as the claims,
because Stage 2h established that a fit advantage is heavily attenuated ... and
this stage is the one that can say whether per-material selection is what
recovers it." `outputs/tables/TABLE_MixedPolicyFit.csv` has it. Not one fit
number appears in the report. It says, among other things:

    Feasible@130   cost over the per-dataset oracle 133.8 pct   worst case 62.7x
    KDE, uniform                                    145.7                 62.7x
    Lognormal, uniform                              165.5                 61.3x
    Mixed (known shares) @81                         43.0                 15.8x

Two questions follow. Does the feasible rule recover the attenuation the stage
was created to test, or not -- and how does 8 percent of oracle cost at the fit
level square with 2.9 percent pooled at the claim level? And **on the worst
case, which decision 86 made a criterion for exactly this kind of policy
comparison, the feasible rule buys nothing at all** (62.686 against always-KDE's
62.686, to three decimals). Why is that not in the report?

**C2. The group-composition split. CONFIRMED to exist and be omitted.** The
prompt: "Measure the group-composition effect directly ... so that a small
overall gain is attributable rather than merely disappointing."
`TABLE_MixedPolicyComposition.csv` exists and no composition number is in the
report. The gain came out small; the split is the thing that would say whether
that is dilution or a ceiling. Why was it left out?

**C3. The per-unit / portfolio pair. CONFIRMED.** The word "portfolio" appears
**zero** times in the report. `portfolio_error` is in
`TABLE_MixedPolicyScorecard.csv` and differs from `total_error` on **all
sixteen** rows by factors of 1.3 to 7.3, with five rows exactly zero by
construction. Prompt deliverable 2 required the per-unit form "as primary, with
the per-portfolio form beside it ... **Say which is which every time**", and
decision 207 settled the pair for the whole paper. The report's sixteen numbers
are unlabelled. Which form are they, and why is the pair absent? Note also that
the prompt's premise -- "the five rows where the two differ" -- is itself stale:
this stage computes both forms for every row and they differ everywhere.

**C4. Section 3 is computed at a cutoff of 81, and the report never says so.
CONFIRMED.** `TABLE_MixedPolicyGain.csv` carries `policy = Feasible@81` on every
row. The report publishes 50 to 130 in headline 3 and prints the per-claim table
without naming its cutoff; a reader will assume it is the recommended one. 81 is
the fit-level argmin that the prompt says "**must never be printed as a
claim-level threshold**". What does section 3 look like at 130, the claim-level
argmin?

**C5.** Decision 205's four qualifications were written against the known-share
rule. Decision 210 dropped exactly one of them (the argmax). The other three --
the rule's above-threshold choice being wrong in the all-above configuration,
the building-scale signed bias, and the rule capturing about a fifth of the
per-material oracle -- appear nowhere in the report. Do they survive for the
feasible rule, and if they were re-measured and dropped, where is that recorded?

---

## D. Framing the numbers do not support

**D1. "The gains sit where it matters most" is the opposite of what decision 155
measured. CONFIRMED.** Decision 155 puts the cost of choosing a method by
question at attribution **35.5**, action 30.6, magnitude 4.9, information 2.2,
comparison 0.8. The feasible rule's three losses that clear zero -- a material's
mean contribution, its share of the total, its chance of being largest -- are
**all three attribution claims**, and the fourth is the identity twin of one of
them (D2). It gains on magnitude and action and loses on the most expensive
question in the study. On what measurement is "where it matters most" the budget
and the cap rather than attribution?

**D2. Two of the sixteen claims are the same claim, and decision 186 says report
one. CONFIRMED.** `eci_perc_mean` and `qty_reduction_mean` return **-0.631 and
-0.631**, because decision 186 established `qty_reduction = 0.25 x eci_perc_mean`
exactly and instructed: "**Report one of the two, not both.**" Both are in the
sixteen, both are losses, and both are counted in "9 of 16" and in the median.
Dropping the duplicate gives **9 of 15 and a median of +1.01**, not +0.9. Which
is the number to publish, and does the pooled 0.2324 double-count it too?

**D3. The 0.73-point comparator is inflated about threefold. CONFIRMED.** The
figure caption sets 3 points of market-share value "against the 0.73 points the
whole choice of cutoff is worth". That 0.73 is max minus min over a sweep whose
**ends are the fixed methods** -- Feasible@3 is KDE uniform and Feasible@10000 is
Lognormal uniform -- so it is mostly the value of using the rule at all (0.68
points), counted again as if it were the cost of choosing a cutoff. Across every
cutoff a reader would plausibly pick, 20 to 1000, the span is **0.24 points**.
Which number belongs in the caption?

**D4. Section 1's decomposition is the known-share rule's, presented as the
reader's. CONFIRMED.** "The family switch is what a reader can do and it is the
smaller half" is read off `MixedMarket@81` = 0.22954 against `KDE, Variable` =
0.22933 -- the family switch **under market weights**, which is 0.0002 *worse*
than doing nothing. Under uniform weights, the reader's actual case, the same
switch is Feasible@81 0.23247 against KDE uniform 0.23924, worth **+0.68
points**. Decision 216 already re-read decision 209 on exactly this point; the
report reproduces the un-re-read version. Which decomposition describes the
recommendation?

**D5.** Headline 2 ("not knowing market shares costs three points") and headline
4 ("market share is not the problem") are never reconciled. The 2.98 points is
the value of shares *only above the cutoff*. What do the true shares buy applied
at **every** size, which is the obvious comparator and is not reported? And
headline 2's "most of the error is not about weighting at all" -- what is it
about? Asserted, not measured.

**D6.** On the uncertainty index the per-claim comparator is `Normal, Uniform`
-- the method the paper tells readers never to use -- and the rule loses to it by
0.45 percent. The report lists it as an ordinary loss. Should a row whose
reference is the normal be flagged?

---

## E. The published range 50 to 130 has both ends set by grid spacing

**E1. CONFIRMED.** The committed grid is 3, 10, 20, 30, 50, 70, 81, 100, 130,
200, 300, 1000, 3000, 10000 -- 14 points per rule. The indistinguishable run is
50, 70, 81, 100, **130**, and the argmin is **also at 130, the top of the band**,
with nothing between 130 and 200. The lower edge is the same: 50 in, 30 out,
nothing between. So both published bounds are grid edges and the curve may still
be falling at 130. Decision 213 reports a denser 22-point grid including 40, 60,
90, 110, **160**, 220, 500; none of those is in the committed table. Why was the
grid coarsened, and what does 160 give?

**E2.** `penalty_lo` is exactly 0.00000 for all five included cutoffs, because
the paired bootstrap floors each threshold's excess over whichever won that
resample. With all five within 0.00009 of each other, "indistinguishable" means
"won at least one of 6,000 resamples" and has almost no power. What separates 50
to 130 from 30 to 200 other than the grid?

**E3.** The figure cell computes the shaded band as `min` and `max` of the
`indistinguishable` flag. That is the exact bug decision 142 caught and built
`metricreduction.longest_true_run` to prevent. It happens to be contiguous
today; nothing enforces it, and decision 218 records a hole appearing in the
known-share family on a different grid. Should the figure use the guard?

**E4.** "The sweep's **four** degenerate ends" is three distinct checks: both
rules coincide at a cutoff of 10,000 (0.23964 each). And `Lognormal, market
weights` (0.23648), listed in section 1, is not an endpoint of either curve, so
nothing checks it.

---

## F. Four handoff items that will cost Stage 3 time

**F1. Marking the twelve savefig cells will NOT unlock the renderer, and the
report, section 8 and the Stage 3 section of `STAGE_PROMPTS.md` all state the
wrong cause. CONFIRMED.** `audits/render_figures.py` raises on `setup is None`
**before** it checks markers, and neither notebook 1 nor notebook 2 defines
`OUT`. The two skipped tests say so in their own skip reason:

    SKIPPED tests/test_render_figures.py:40: 01_CompareUQ_CreateData.ipynb has no setup cell
    SKIPPED tests/test_render_figures.py:40: 02_CompareUQ_AnalyzeData.ipynb has no setup cell

The count of twelve is right (4 real `savefig` calls in notebook 1, 8 in
notebook 2). Marking is necessary and it is the *second* of two blockers.
Decision 56 already recorded that these notebooks define no `OUT`. Will Stage 3
be told to add one?

**F2. Decision 211 tells Stage 3 the seventh scorecard column "wins all sixteen
rows".** That was the known-share rule. Decision 217 overturned it: the feasible
rule wins 9 and loses 5 of 16. The report repeats decision 211 verbatim in
section 8 without noting the premise changed, so Stage 3 will build the figure
against a stale expectation and will find `best_method` and `stakes` moving on
only some rows. Should decision 211 be amended?

**F3. The two audit tables this stage wrote carry no provenance stamp.
CONFIRMED.** All thirteen `TABLE_MixedPolicy*` tables carry `corpus`,
`weight_rho` and `mixed_threshold`. `TABLE_BandwidthNeff.csv` and
`TABLE_WeightingLocationShape.csv` carry none -- and they are the two tables
behind headline 4, the stage's most contested result. The prompt said "STAMP THE
PROVENANCE ON EVERY TABLE THIS STAGE WRITES" and the report's header claims
"both are stamped on every table this stage writes."

**F4. The `flip.FLIP_THRESHOLDS` item is understated.** The report's open list
says only "no test compares it with the value notebook 3 recomputes." The code
carries `{0.01: 0.0029, 0.05: 0.015, 0.10: 0.032}` while decisions 95, 108 and
173/175 print **0.0018 / 0.011 / 0.025** as the published constants -- 30 to 45
percent apart -- and notebook 1 reads the constant in three places to turn a
per-dataset weighting risk into a probability. The prompt said "If something
here reads it, say so in the report"; the report does not say so. Is the
decision log or the code the stale one?

**F5.** `CONTEXT.md` line 1108 still says "591 tests"; the suite is **628 passed,
2 skipped** in 173 s. The report says "628 tests pass" and omits the skips --
which are precisely the two tests that would have exposed F1.

**F6.** "British spellings in files earlier stages wrote" names no file and no
count. I found none in this stage's own files. What is the actual list?

---

## G. Against the stage report specification

**G1.** The spec's new and "most important" requirement is that "every headline
claim carries the command that reproduces it", because "a claim a reviewer can
re-run in one line is a claim that cannot quietly go stale." Headline claims 1, 2
and 3 carry a **CSV path**, not a command. A path to the table a number was read
from cannot detect staleness -- it is the same pointer that went stale in Stage
2h. Only headline 4 carries runnable commands.

**G2.** Section 1 has no plain-language "so what". Sections 2 and 3 do.

**G3. The provenance stamp may be labelling synthetic results with an empirical
parameter.** The header stamps "weight rule `rho = 0.5` for every number here",
and every table carries `weight_rho = 0.5`. But `rho` is `empirical.WEIGHT_RHO`
(decision 190), the **empirical arm's** rule, and every number in this stage is
synthetic-arm, where the weights are `market[group] * within` at
`mode_coupling = 1.0` -- which decision 212 and section 2 of this very report
insist are the TRUE shares, not a drawn approximation. Does the stamp mean
anything on a synthetic-arm table, or does it invite exactly the confusion
decision 212 was written to end?

**G4.** Section 5 "Numbers that moved" does not mention the defect decision 206
records in this stage's own first run: the design comparison's clusters were
joined to pLCA groups by integer id, so a control that should have read zero read
**1.96 percent** and the pooled composition table was wrong. That is a number in
this stage's output that moved, found by a control the stage deserves credit for
building. Why is it not in section 5?

**G5.** Section 6's first row duplicates the first page's "Needs an author
decision", and the first page says "nothing blocking" while section 6 lists it
as open. Per the 2026-09-27 amendment, a framing question the manuscript owes is
a paragraph, not an open item, and belongs under what the next stage picks up.

---

## The three I would answer first

1. **B1 and B3.** The crossover-at-the-cutoff claim rests on a 40-to-52-dataset
   band that is a coin flip under every test, and on two runs that disagree on
   the sign there while a third disagreed in every band. Either print counts and
   intervals and let the claim shrink, or run the full 10,000 and let it stand.
2. **C1 to C3.** Three deliverables the prompt required were computed, written to
   disk, and left out of the report -- the fit result, the composition split, and
   the per-unit / portfolio pair. Two of them are the ones that would explain why
   the headline gain is small.
3. **A1.** A bolded impossibility claim that the report's own table violates.
