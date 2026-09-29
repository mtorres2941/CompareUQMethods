# Stage 2j: let the method vary by material

**Branch** `stage-2j-per-material`. **Corpus** `corpus_2026-09-25`, **weight
rule** `rho = 0.5`, for every number here; both are stamped on every table this
stage writes. **Run:** 2,500 pLCA groups against the true parents, 6 fixed
methods plus 31 candidate rules on one set of variates.

---

## What this found

**1. The rule a practitioner can actually follow is a SMALL improvement, not a
large one.** Uniform weights throughout, kernel estimate above the cutoff and
three-parameter lognormal below, against the best of the three uniform-weighted
methods: **+2.9 percent** pooled over the sixteen claims, and claim by claim it
beats the best fixed method for that claim on **9 of 16**, median **+0.9
percent**, range -0.6 to +6.4. Five of the seven it loses carry intervals that
exclude zero, all of them between -0.25 and -0.63 percent.

**2. THE LARGE GAIN NEEDS MARKET SHARES, WHICH NOBODY HAS.** The same rule with
the TRUE market shares above the cutoff beats the best of all six fixed methods
on **16 of 16** claims, median **+11.4 percent**. That is not a method -- it is
**the value of knowing market share, measured at 12.8 percent** of the pooled
error. This report previously led with that number as though it were a
recommendation. It is not one.

**3. Where the cutoff sits barely matters, for either rule.** Swept from 3 to
10,000 declarations. For the feasible rule everything from **50 to 130** is
statistically indistinguishable from the best and the whole sweep spans only
**0.73 points**; for the known-share rule it is **50 to 100** and 3.71 points.
**Round it: switch somewhere between about 50 and 100 declarations.** The
sweep's four degenerate ends reproduce their fixed methods to 0.00e+00, which
is the check that the machinery is right.

**4. Using a market share you KNOW hurts below about 81 declarations, and the
reason is not a bad guess.** These are importance weights on a fixed set of n
observed products, not frequency weights: they re-aim the information you have
at the population that gets built, they do not add any. Below the cutoff that
re-aiming costs more variance than the information is worth. Section 2.

**Needs an author decision:** whether the paper leads with a 0.9 percent median
improvement. It is real and it is small, and it is the honest version of what
was a 11.4 percent claim.

---

## 1. Two rules, and only one of them is a method

    policy                                            pooled error over 16 claims
    FEASIBLE: uniform weights, kernel above the cutoff, lognormal below   0.2324
      KDE, uniform weights                                                0.2392
      Lognormal, uniform weights                                          0.2396
      Normal, uniform weights                                             0.3144
    KNOWN SHARES: the same switch, market weights above the cutoff        0.2025
      KDE, market weights                                                 0.2293
      Lognormal, market weights                                           0.2365

**The feasible rule is worth 2.9 percent over the best method a reader can
choose between. Knowing market shares on top would be worth a further 12.8
percent** -- four times as much as the rule itself. The paper's strongest
practical statement is therefore about the value of market-share data, not
about which curve to fit.

**Which half of the known-share switch does the work** confirms it. Holding one
axis and switching the other: weighting only, 0.2092 and 0.2096; family only
(market weights throughout), 0.2295, which is the best fixed method to three
decimals. **The family switch is nearly all of what the feasible rule has, and
it is the smaller half.**

    Reproduce: outputs/tables/TABLE_MixedPolicyRanking.csv

## 2. What ignoring a KNOWN market share costs

**These are the true shares, not a guess**, and the previous version of this
report said otherwise. At the shipped generator setting each synthetic point
carries `market[group] * within`, so the weight mass on every product group
equals that group's true market share to 1.1e-16.

Share of datasets on which the fit using those true shares is closer to the
truth than the same fit with every declaration weighted equally:

    declarations       3-9   10-80   81-99  100-999   1000+
    lognormal         32.8    42.5    53.6     64.6    78.7
    kernel            38.5    44.1    52.7     59.0    76.2

It crosses half at the cutoff, for both families, with nothing tuned to put it
there.

**Why knowing more can help less.** These are importance weights on a fixed set
of observed products, not frequency weights. You hold n EPDs whatever the
weights say; the weights re-aim that fixed information at the population that
gets built. Market weighting is unbiased for that population and high variance;
uniform weighting is biased -- it estimates the population that publishes -- and
low variance. Below the cutoff the variance wins, above it the bias does. The
Kish effective sample size is that variance cost: under the true weights its
median is **2.8** at 3 to 9 declarations, with **92.4 percent** of such datasets
left below five effective observations, against **33.5** at 81 to 99. Nine EPDs
where one group holds 90 percent of the market and two of the nine leaves the
estimate resting on two observations, however well the share is known.

**And the bandwidth is not the cause**, which had to be checked because the KDE
bandwidth uses the effective sample size. Two arguments: the same crossover
appears in the lognormal, **which has no bandwidth at all**; and refitting the
kernel estimate with the plain count over 2,000 datasets moves the crossover
nowhere -- market-closer shares of 38.9, 46.4, 56.8, 62.4, 77.4 percent against
40.1, 46.2, 61.4, 61.6, 77.2. The uniform column is bit-identical between the
two, which is the internal control.

    Reproduce: outputs/tables/TABLE_MixedPolicyWeighting.csv
               python audits/bandwidth_neff.py --n 2000

## 3. What the feasible rule buys, claim by claim

Against the best of the three uniform-weighted methods on each claim -- a
harder test than the pooled comparison, because the comparator changes per
claim -- with a paired cluster bootstrap over pLCA groups:

    the chance of meeting a budget                 +6.42   [ 5.61,  7.23]
    a cap's chance of saving 5 pct                 +2.81   [ 1.61,  3.93]
    a material's 95th percentile                   +2.73   [ 2.17,  3.30]
    the total's 90th percentile                    +2.33   [ 0.79,  3.89]
    how often a cap binds                          +2.22   [ 1.05,  3.39]
    the total's mean                               +2.17   [ 1.68,  2.69]
    the total's standard deviation                 +1.95   [ 0.66,  3.37]
    a material's share at the building's 95th      +1.01   [ 0.24,  1.79]
    a cap's mean saving                            +0.82   [ 0.46,  1.18]
    ---- above this line the interval clears zero ----
    a material's standard deviation                +0.53   [-0.38,  1.43]
    the probability B beats A                      +0.19   [-0.51,  0.90]
    a material's mean contribution                 -0.25   [-0.51, -0.01]
    the uncertainty index                          -0.45   [-1.67,  0.75]
    a material's chance of being largest           -0.53   [-0.97, -0.07]
    a material's share of the total                -0.63   [-0.83, -0.44]
    using 25 pct less: its mean saving             -0.63   [-0.82, -0.43]

**Nine of sixteen, median +0.9 percent, and four of the seven losses are
statistically real though all are under two thirds of a percent.** The gains
are concentrated in the tail and intervention claims -- meeting a budget, what
a specification cap delivers, the 90th and 95th percentiles -- and the losses
in the mean-and-share claims, where a lognormal fitted to every material is
already about as good as anything.

**So what.** Switching family by size buys a practitioner a little accuracy
where it matters most -- budget compliance and what an intervention delivers --
and costs a little on the averages. It is worth doing and it is not
transformative.

    Reproduce: outputs/tables/TABLE_MixedPolicyGain.csv, rows where rule=feasible

## 4. The figure

![Mean error over the 16 claims against where the method switches](../outputs/figures/CompareUQMethods_FIG_MixedPolicy.png)

Mean error over the sixteen claims a probabilistic LCA makes, each divided by
its own true level, against the cutoff on a log axis. **The orange curve is the
rule a reader can follow**; the grey one adds the true market shares above the
cutoff and is not a method. **The vertical gap between them is what knowing
market share would be worth** -- about three points of the roughly twenty-three
that a probabilistic LCA gets wrong, against the 0.73 points the whole choice of
cutoff is worth. The two dotted lines are fixed methods; the curves' four ends
land on them exactly, because at a cutoff of 3 or 10,000 each rule IS a fixed
method. The shaded band is the cutoffs indistinguishable from the best.

*If the image does not render:* two shallow U curves on a log x axis from 3 to
10,000 declarations. The upper, orange, runs from 23.92 percent at a cutoff of
3 down to 23.24 at 130 and back to 23.96 at 10,000. The lower, grey, runs from
22.93 down to 20.25 at 70 and back to 23.96. A shaded band covers 50 to 130.

## 5. Numbers that moved

**In the study's existing tables: none.** The committed notebook was run end to
end with no error in any cell and every pre-existing table reproduces
content-identically; the only differing bytes are gzip header timestamps and
one `written_utc` field.

**In this stage's own earlier output: the headline.** The previous version of
this report led with "beats every fixed method on all sixteen claims, median
11.4 percent". That number is unchanged and it belongs to the known-share rule,
which requires information nobody has. The feasible rule's median is **0.9
percent**. Nothing was recomputed to get there; what changed is which rule the
report calls the recommendation.

**One change to shared code.** `plca.swap_run` drew the same five columns once
per claimed saving; the draws are now hoisted out of that loop. Bit-identical,
pinned by a test, and it is what made a 37-column sweep affordable.

## 6. Still open

| Item | |
|---|---|
| **Whether the paper leads with a 0.9 percent median improvement**, and how it frames the 12.8 percent value of market-share data beside it |
| **Whether to round the cutoff range to 50-100** (both rules' ranges contain it; the feasible rule tolerates up to 130) |
| **Every figure except the scorecard and this one still carries the retired weighting labels.** Twelve savefig cells across notebooks 1 and 2 are unmarked, so the fast renderer refuses those notebooks; marking them is Stage 3's first task and the prerequisite for the sweep either way |
| **No test compares `flip.FLIP_THRESHOLDS` with the value notebook 3 recomputes.** Stage 3 |
| **British spellings in files earlier stages wrote.** The deposit tidy-up |

## 7. Inputs, outputs, reproduce

**Read:** the synthetic corpus and the true parents replayed from it;
`TABLE_MethodScores.csv` from notebook 2.

**Written:** thirteen tables named `TABLE_MixedPolicy*` plus
`CompareUQMethods_FIG_MixedPolicy.png`. **Code:** `src/mixedpolicy.py`,
`tests/test_mixedpolicy.py` (36 tests), `audits/bandwidth_neff.py`, the hoist in
`src/plca.py`, and nine cells at the end of notebook 3. **628 tests pass.**

    cd notebooks && python -m nbconvert --to notebook --execute \
      --ExecutePreprocessor.kernel_name=compareuq \
      --output-dir=/tmp/nbrun --output=out.ipynb 03_CompareUQ_PerformPLCA.ipynb

About 95 minutes, of which the Stage 2j block is about 35. The figure alone:

    python audits/render_figures.py 03_CompareUQ_PerformPLCA \
      --only "how much the cutoff matters" --into-outputs

## 8. What Stage 3 picks up first

1. **Add the feasible rule as a seventh column to the scorecard figure**
   (decision 211) and fold this stage's per-claim gains into it. Use the
   FEASIBLE rule, not the known-share one: a scorecard column a reader cannot
   reproduce is worse than none.
2. **Mark the twelve unmarked savefig cells in notebooks 1 and 2**, which
   unlocks the fast renderer, then do the caption sweep: the retired weighting
   labels and the corrected weighting framing of decision 212.
