# Stage 2j: let the method vary by material

**Branch** `stage-2j-per-material`. **Corpus** `corpus_2026-09-25`, **weight
rule** `rho = 0.5`, for every number here; both are stamped on every table this
stage writes. **Run:** 2,500 pLCA groups against the true parents, 6 fixed
methods plus 31 candidate rules on one set of variates.

---

## What this found

**1. The rule a practitioner can follow buys a little.** Uniform weights
throughout, kernel estimate at or above the cutoff and three-parameter
lognormal below, against the best of the three uniform-weighted methods:
**+2.9 percent** pooled over the sixteen claims, and claim by claim it beats
the best fixed method for that claim on **9 of 16**, median **+0.9 percent**.
The gains sit where it matters most -- the chance of meeting a budget at +6.4
percent, what a specification cap delivers at +2.8 -- and the losses, four of
them statistically real, are all under two thirds of a percent.

**2. NOT KNOWING MARKET SHARES COSTS ABOUT THREE POINTS OUT OF TWENTY-THREE.**
Give the same rule the true market shares above the cutoff and the pooled error
falls from **23.24 to 20.25 percent** of the true level: **2.98 points, or 12.8
percent of what was there.** The relative number sounds large and the absolute
one is the honest frame -- a probabilistic LCA is wrong by about 23 percent
either way, and knowing every market share exactly would take it to 20. Most of
the error is not about weighting at all.

**3. Where the cutoff sits barely matters.** Swept 3 to 10,000 declarations.
For the feasible rule **everything from 50 to 130 is statistically
indistinguishable** from the best and the whole sweep spans 0.73 points.
**Publish 50 to 130.** The sweep's four degenerate ends reproduce their fixed
methods to 0.00e+00, which is the check that the machinery is right.

**4. The bandwidth is not why market weighting loses at small n, and this is
now settled three ways.** Section 2.

**Needs an author decision:** nothing blocking. The judgment left is how the
paper frames a 0.9 percent method improvement beside a 3-point value of
market-share data.

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

**Which half of the known-share switch does the work.** Holding one axis and
switching the other: weighting only, 0.2092 and 0.2096; family only, 0.2295,
which is the best fixed method to three decimals. The family switch is what a
reader can do and it is the smaller half.

    Reproduce: outputs/tables/TABLE_MixedPolicyRanking.csv

## 2. What NOT knowing a market share costs

**These are the true shares, not a guess.** At the shipped generator setting
each synthetic point carries `market[group] * within`, so the weight mass on
every product group equals that group's true market share to 1.1e-16.

Share of datasets on which a fit that knows those shares is closer to the truth
than the same fit with every declaration weighted equally:

    declarations       3-9   10-80   81-99  100-999   1000+
    lognormal         32.8    42.5    53.6     64.6    78.7
    kernel            38.5    44.1    52.7     59.0    76.2

**Below about 81 declarations, knowing the shares does not help.** It crosses
half at the cutoff, for both families, with nothing tuned to put it there.

### Why, and it is not the effective sample size in the bandwidth

**The earlier explanation in this report was sloppy and the objection to it was
right.** Nine EPDs are nine EPDs; learning the market shares does not take
observations away. What changes is the QUESTION. Nine declarations give you
nine observations of the population that PUBLISHES. Learning the shares tells
you that seven of those nine describe a product group that is ten percent of
what actually gets BUILT -- so for the market-weighted question your sample is
badly aimed, and it always was. The effective sample size is not a penalty the
method imposes; it is how many of your nine are pointed at the question you now
know you are asking.

**And the consequence is a bias-variance trade with a floor.** Each fit given
the bandwidth that minimizes its own distance to the truth, mean W1 against the
true market-weighted parent:

    declarations            3-9   10-80   81-99  100-999   1000+
    uniform weights      0.2621  0.1614  0.1142   0.0968  0.0844
    known market shares  0.3005  0.1808  0.1361   0.0762  0.0287
    how far the two populations differ  ~0.10 to 0.12, at every size

**The market-weighted fit converges: 0.30 to 0.029, still falling. The
uniform-weighted fit does not: it flattens at 0.084 against a floor of 0.099,
which is the distance between the population that publishes and the population
that gets built.** No quantity of EPDs gets a uniform-weighted fit below that
floor. Below 81 declarations its variance advantage is bigger than the floor;
above it, it is not.

### The bandwidth, settled three ways

The kernel bandwidth uses the effective sample size, so a concentrated weight
vector widens it. Three tests, each stronger than the last -- share of datasets
on which the market-weighted kernel estimate is closer:

    declarations                        3-9   10-80   81-99  100-999   1000+
    production rule, n_eff             40.3    44.0    53.8     61.4    77.9
    the plain count n                  38.8    43.7    50.0     61.9    78.4
    each fit's OWN BEST bandwidth      46.0    42.7    51.9     55.3    74.6

**Using the plain count does not flip it.** **And giving each fit the bandwidth
that minimizes its own distance to the truth -- which no rule can beat -- does
not flip it either**: the market-weighted fit still loses on 54 percent of
datasets at 3 to 9 declarations. The bandwidth costs it about 6 points there
and the crossover is still below 81 without it.

**A fourth argument needs no bandwidth at all**: the three-parameter lognormal
has none, and it shows the same crossover, 32.8 percent at 3 to 9 and 78.7
above a thousand.

    Reproduce: python audits/bandwidth_neff.py --n 2500

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

**So what.** Switching family by size buys accuracy on the tail and
intervention claims -- budget compliance, what a cap delivers -- and costs a
little on the averages, where a lognormal fitted to everything is already about
as good as anything. Worth doing, not transformative.

    Reproduce: outputs/tables/TABLE_MixedPolicyGain.csv, rows where rule=feasible

## 4. The figure

![Mean error over the 16 claims against where the method switches](../outputs/figures/CompareUQMethods_FIG_MixedPolicy.png)

Mean error over the sixteen claims a probabilistic LCA makes, each divided by
its own true level, against the cutoff on a log axis. **The orange curve is the
rule a reader can follow.** The grey one adds the true market shares above the
cutoff and is not a method; **the vertical gap between them is what knowing
market share would be worth -- about 3 points of the roughly 23 a probabilistic
LCA gets wrong**, against the 0.73 points the whole choice of cutoff is worth.
The dotted lines are fixed methods, and the curves' four ends land on them
exactly, because at a cutoff of 3 or 10,000 each rule IS a fixed method. The
shaded band is 50 to 130, the cutoffs indistinguishable from the best.

*If the image does not render:* two shallow U curves on a log x axis from 3 to
10,000 declarations. The upper, orange, runs from 23.92 percent at a cutoff of
3 down to 23.24 at 130 and back to 23.96 at 10,000. The lower, grey, runs from
22.93 down to 20.25 at 70 and back to 23.96. A shaded band covers 50 to 130.

## 5. Numbers that moved

**In the study's existing tables: none.** The committed notebook was run end to
end with no error in any cell and every pre-existing table reproduces
content-identically; the only differing bytes are gzip header timestamps and
one `written_utc` field.

**In this stage's own earlier output: the headline.** An earlier version of
this report led with "beats every fixed method on all sixteen claims, median
11.4 percent". That number is unchanged and belongs to the known-share rule,
which needs information nobody has. The feasible rule's median is **0.9
percent**. Nothing was recomputed; what changed is which rule is called the
recommendation.

**One change to shared code.** `plca.swap_run` drew the same five columns once
per claimed saving; the draws are now hoisted out of that loop. Bit-identical,
pinned by a test, and it is what made a 37-column sweep affordable.

## 6. Still open

| Item | |
|---|---|
| **How the paper frames a 0.9 percent method improvement beside a 3-point value of market-share data.** Both are real; the second is the larger story and it is an argument for obtaining production volumes |
| **Every figure except the scorecard and this one still carries the retired weighting labels.** Twelve savefig cells across notebooks 1 and 2 are unmarked, so the fast renderer refuses those notebooks; marking them is Stage 3's first task and the prerequisite either way |
| **No test compares `flip.FLIP_THRESHOLDS` with the value notebook 3 recomputes.** Stage 3 |
| **British spellings in files earlier stages wrote.** The deposit tidy-up |

## 7. Inputs, outputs, reproduce

**Read:** the synthetic corpus and the true parents replayed from it;
`TABLE_MethodScores.csv` from notebook 2.

**Written:** thirteen tables named `TABLE_MixedPolicy*` plus
`CompareUQMethods_FIG_MixedPolicy.png`, and
`outputs/tables/audits/TABLE_BandwidthNeff.csv`. **Code:**
`src/mixedpolicy.py`, `tests/test_mixedpolicy.py` (36 tests),
`audits/bandwidth_neff.py`, the hoist in `src/plca.py`, and nine cells at the
end of notebook 3. **628 tests pass.**

    cd notebooks && python -m nbconvert --to notebook --execute \
      --ExecutePreprocessor.kernel_name=compareuq \
      --output-dir=/tmp/nbrun --output=out.ipynb 03_CompareUQ_PerformPLCA.ipynb

About 95 minutes, of which the Stage 2j block is about 35. The figure alone:

    python audits/render_figures.py 03_CompareUQ_PerformPLCA \
      --only "how much the cutoff matters" --into-outputs

## 8. What Stage 3 picks up first

1. **Add the FEASIBLE rule as a seventh column to the scorecard figure**
   (decision 211) and fold this stage's per-claim gains into it. Not the
   known-share rule: a scorecard column a reader cannot reproduce is worse
   than none.
2. **Mark the twelve unmarked savefig cells in notebooks 1 and 2**, which
   unlocks the fast renderer, then do the caption sweep: the retired weighting
   labels and the corrected framing of decisions 212 and 219.
