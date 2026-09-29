# Stage 2j: let the method vary by material

**Branch** `stage-2j-per-material`. **Corpus** `corpus_2026-09-25`, **weight
rule** `rho = 0.5`, for every number here; both are stamped as columns on every
table this stage writes. **Run:** 2,500 pLCA groups against the true parents, 6
fixed methods plus 26 candidate policies on one set of variates.

---

## What this found

**1. Choosing the method material by material beats every fixed method on all
sixteen claims a probabilistic LCA makes**, by a median **11.4 percent** of the
best fixed method's own error, every paired interval clearing zero. Pooled over
the sixteen claims it is **0.2026** against **0.2293** for the best fixed
method.

**2. Almost any cutoff works.** Swept from 3 to 10,000 declarations: **every
cutoff from 5 to 3,000 beats both fixed methods**, and **50 to 81 are
statistically indistinguishable from the best**, which is 70. The two ends of
the sweep are a built-in check -- the corpus holds 3 to 9,999 declarations, so
a cutoff of 3 IS always-kernel-with-market-weights and a cutoff of 10,000 IS
always-lognormal-with-uniform-weights, and both reproduce those methods to
0.00e+00.

**3. THE GAIN IS THE WEIGHTING SWITCH, NOT THE FAMILY SWITCH.** The rule
switches two things at one cutoff; holding one and switching the other
separates them:

    policy                                              pooled error over 16 claims
    the rule: kernel + market above, lognormal + uniform below   0.2026
    kernel throughout, weighting switches at the cutoff          0.2092
    lognormal throughout, weighting switches at the cutoff       0.2096
    best fixed method (kernel, market weights)                   0.2293
    market weights throughout, family switches at the cutoff     0.2295
    uniform weights throughout, family switches at the cutoff    0.2325

**Switching only the WEIGHTING recovers three quarters of what the rule buys.
Switching only the FAMILY recovers nothing** -- it lands on the best fixed
method. The family switch is worth a further 3 percent on top.

**4. A CORRECTION, AND IT IS THE IMPORTANT ITEM ON THIS PAGE.** The previous
version of this report said the study's market weights are "a flat-Dirichlet
guess". **That is false, and the objection to it was right.** On the synthetic
arm -- which is every number in this stage -- the weight mass sitting on each
product group EQUALS that group's true market share, to 1.1e-16; only the
division of a group's share among the products inside it is arbitrary. So the
comparison here is **ignoring a market share you know against using it**, which
is the question that was being asked. Section 2 has the corrected result and
its mechanism. No number moves; the words around them were wrong, and they had
been wrong since Stage 2e.

**Needs an author decision:** whether the paper prints the indistinguishable
range as **50 to 81** (measured) or rounds it. Nothing else is blocking.

---

## 1. Where to put the cutoff barely matters

Twenty-two cutoffs from 3 to 10,000, each scored on all sixteen claims against
the true parents, with the same instrument the fit-level work uses: two
bootstraps over pLCA groups, the second PAIRED against whichever cutoff won on
that same resample, at 6,000 resamples.

    cutoff       3     5    10    20    40    50    60    70    81    90   100
    pooled   .2293 .2221 .2131 .2073 .2037 .2028 .2029 .2025 .2026 .2029 .2028
    cutoff     130   220   300   500  1000  3000 10000
    pooled   .2037 .2058 .2075 .2109 .2161 .2263 .2396

**The measured answer is 50 to 81**, the longest unbroken run of cutoffs whose
paired interval reaches zero, with the minimum at 70. Read it with the shape
beside it: 100 is individually indistinguishable but 90 is not, so the unbroken
run stops at 81, and everything from 40 to 130 sits within 0.0011 of the best.
**And the practical statement is wider: every cutoff from 5 to 3,000 beats both
fixed methods.** Across the whole sweep the cutoff moves the pooled error by
3.71 points, of which 2.68 is the rule beating the best fixed method and the
rest is the choice of where to put it.

**So what.** A practitioner who switches method somewhere between fifty and a
hundred declarations gets the best available answer, and one who switches
anywhere between five and three thousand still does better than sticking to one
method. Getting the number right is worth far less than having the rule.

## 2. What ignoring a KNOWN market share costs

**First, what the weights are**, because the previous report described them
wrongly. At the shipped generator setting each synthetic point carries
`market[group] * within`, where `market` is the true market share the parent
was built with. The weight mass on each product group therefore equals that
group's true market share exactly. Nobody is guessing anything; the only
arbitrary part is how a group's share is split among the products inside it.

Share of datasets on which the fit using those true market shares is closer to
the truth than the same fit with every declaration weighted equally:

    declarations       3-9   10-80   81-99  100-999   1000+
    lognormal         32.8    42.5    53.6     64.6    78.7
    kernel            38.5    44.1    52.7     59.0    76.2

**Using a market share you know HURTS below about 81 declarations.** It crosses
half at the cutoff, for both families, with nothing tuned to put it there.

**The mechanism is the effective sample size.** Under those true weights the
Kish effective sample size has a median of **2.8** at 3 to 9 declarations, with
**92.4 percent** of such datasets left below five effective observations,
against **33.5** at 81 to 99 and **1,099** above a thousand. A concentrated
market share spends your sample: below the cutoff that costs more than the
information is worth, above it the information wins.

**So what.** Market share is real information and using it is right when you
have enough declarations to afford it. With nine EPDs, weighting by market
share leaves you estimating a distribution from about three effective points,
and you are better off treating them equally.

**One thing the paper must keep apart.** Real EC3 categories have no published
market shares, so the empirical arm simulates them. That arm can say what
weighting WOULD do under a plausible share model; it cannot say what ignoring a
known share costs. Every number in this stage is synthetic, so this stage does
answer that question.

## 3. What the rule buys, claim by claim

Every gain is against the best of the six FIXED methods on that claim -- the
comparator a reader would otherwise use -- with a paired cluster bootstrap over
pLCA groups. **All sixteen are positive and all sixteen clear zero**, median
11.4 percent:

    how often a specification cap binds               +16.8   [14.6, 19.0]
    a cap's chance of saving 5 pct of the building    +15.4   [13.3, 17.6]
    the chance of meeting a budget                    +15.1   [11.6, 18.8]
    a material's chance of being largest              +14.3   [13.1, 15.6]
    ...
    a material's standard deviation                    +5.7   [ 4.2,  7.0]
    the uncertainty index                              +5.2   [ 3.5,  7.0]
    a material's share at the building's 95th pct      +3.3   [ 1.1,  5.6]

On the design comparison the rule is wrong about which design is better on
**17.8 percent** of individual comparisons at a claimed 5 percent saving,
against 19.8 for the best fixed method, and its per-pair error is 0.0713
against 0.0823.

**So what.** The claims that move most are the ones about interventions --
whether a specification cap binds and what it delivers -- which is where a
design team acts.

## 4. Where the rule is not the best choice

**It is the most accurate policy per building and not the least biased.** Its
absolute error on one building's mean total is **8.49 percent** of the true
total, the lowest of the seven, and its signed bias is **-2.86 percent** against
**+0.91** for a kernel estimate with market weights everywhere. Bias adds across
the materials of a building while noise cancels, so for one building follow the
rule and for a portfolio or a stock model a kernel estimate with market weights
everywhere is safer.

**It captures about a fifth of what a per-material choice could buy.** Against
an oracle that picks whichever of the six is closest on each unit -- which needs
the answer in order to choose, and is a minimum over six noisy errors -- the
rule closes a median **22.5 percent** of the distance.

**The group is what limits it**, and the split carries its own control. Pooled
error by how many of a group's four materials the rule moves:

    materials moved      0      1      2      3      4
    groups             119    573    949    687    172
    the rule          .3260  .2730  .2168  .1505  .0842

At 0 and at 4 the rule IS a fixed policy and matches it to 0.00e+00 on every
output. Everything it buys is in the middle, and 88.4 percent of groups are in
the middle.

## 5. The figure

![Mean error over the 16 claims against where the method switches](../outputs/figures/CompareUQMethods_FIG_MixedPolicy.png)

Mean error over the sixteen claims a probabilistic LCA makes, each divided by
its own true level, against the cutoff on a log axis. The two dotted lines are
the fixed methods the rule switches between, at 22.9 and 24.0 percent. **The
curve's ends land on them exactly**, because a cutoff of 3 assigns every dataset
the kernel estimate and a cutoff of 10,000 assigns every dataset the lognormal
-- that is the sweep's self-check, not a coincidence. The shaded band is the
cutoffs whose paired interval reaches zero.

**One panel, not two.** The per-claim gains that were the left panel belong in
the scorecard figure, where they can be read against how accurate each method
is; Stage 3 folds them in. This panel says the thing the scorecard cannot.

*If the image does not render:* a shallow U falling from 22.9 percent at a
cutoff of 3 to 20.3 at 70 and rising to 24.0 at 10,000, with a shaded band over
50 to 81 and two horizontal reference lines at 22.9 and 24.0 that the curve's
two ends sit on.

## 6. Numbers that moved

**In the study's existing tables: none.** The committed notebook was run end to
end with no error in any cell and every pre-existing table reproduces
content-identically; the only differing bytes are gzip header timestamps and
one `written_utc` field. That is by construction: the Stage 2j cells sit at the
end, consume no randomness before any existing cell, and run their own truth
pass rather than extending the study's.

**No number moved for the weighting correction either.** Decision 212 changes
what the measurements mean, not what they are.

**One change to shared code.** `plca.swap_run` drew the same five columns once
per claimed saving; the draws are now hoisted out of that loop. Bit-identical --
a test asserts it and the committed six-method design-swap table reproduces
content-identically -- and it is what made a 32-column sweep affordable.

## 7. Still open

| Item | |
|---|---|
| **Whether the paper prints the indistinguishable range as 50 to 81 or rounds it.** Measured: best 70, unbroken indistinguishable run 50 to 81, within 0.0011 of best from 40 to 130, and every cutoff from 5 to 3,000 beating both fixed methods |
| **Whether the recommendation is stated as a weighting switch with a family switch on top**, which is what item 3 of the first page measures, rather than as one two-part rule |
| **Every figure except the scorecard and this one still carries the retired weighting labels**, and the scorecard cell's comment block still says "sampled market shares". Stage 3's caption sweep |
| **No test compares `flip.FLIP_THRESHOLDS` with the value notebook 3 recomputes.** Stage 3 |
| **British spellings in files earlier stages wrote.** The deposit tidy-up |

**Dropped since the last report:** the argmax qualification. The study does not
report that metric -- decisions 102, 143 and 155 demoted it three times -- and
carrying it as a mark against the rule gave a retired metric a vote. The number
is in `TABLE_MixedPolicyTruth.csv.gz` for anyone who asks.

## 8. Inputs, outputs, reproduce

**Read:** the synthetic corpus and the true parents replayed from it;
`TABLE_MethodScores.csv` from notebook 2.

**Written:** thirteen tables named `TABLE_MixedPolicy*` -- the threshold curve,
the policy ranking, the per-claim gain, the fit, the weighting crossover with
its effective sample sizes, the scorecard, the ceiling, the composition split
and the four row-level truth frames -- plus
`CompareUQMethods_FIG_MixedPolicy.png`.

**Code:** `src/mixedpolicy.py` and `tests/test_mixedpolicy.py` (34 tests); the
hoist in `src/plca.py` and its test; nine cells at the end of notebook 3.
**626 tests pass.**

    cd notebooks && python -m nbconvert --to notebook --execute \
      --ExecutePreprocessor.kernel_name=compareuq \
      --output-dir=/tmp/nbrun --output=out.ipynb 03_CompareUQ_PerformPLCA.ipynb

About 90 minutes, of which the Stage 2j block is about 30. The figure alone
re-renders in seconds:

    python audits/render_figures.py 03_CompareUQ_PerformPLCA \
      --only "how much the cutoff matters" --into-outputs

## 9. What Stage 3 picks up first

1. **Add the rule as a seventh column to the scorecard figure** (decision 211),
   and fold this stage's per-claim gains into it. It changes `best_method` on
   all sixteen rows and `stakes` on every row, so every sentence about which of
   six methods is best has to be re-read against it.
2. **The caption sweep**, including the retired weighting labels everywhere
   except the two figures that use `display_method` -- and the corrected
   weighting framing of decision 212, which affects prose rather than figures.
