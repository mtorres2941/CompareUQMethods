# Stage 2j: let the method vary by material

**Branch** `stage-2j-per-material`. **Corpus** `corpus_2026-09-25`, **weight
rule** `rho = 0.5`, for every number here; both are stamped as columns on
every table this stage writes. **Run:** 2,500 pLCA groups against the true
parents, 6 fixed methods plus 17 candidate policies on one set of variates.

---

## What this found

**1. Choosing the method material by material beats every fixed method on all
sixteen claims a probabilistic LCA makes**, by a median **11.4 percent** of the
best fixed method's own error, every paired interval clearing zero. Pooled over
the sixteen claims it is **0.2026** against **0.2293** for the best fixed
method, a gain of 11.7 percent.

**2. The cutoff barely matters, which is the answer to "how sensitive is it".**
Across the whole swept range of 20 to 220 declarations the pooled error moves
by **0.47 points** on a level of about 20, against **2.68 points** for the rule
itself over the best fixed method. The best cutoff is **70**; everything from
**50 to 81** is statistically indistinguishable from it, and no cutoff between
40 and 130 costs more than 0.11 points. **The paper should publish a range of
roughly 50 to 100 declarations and not a point.** The fit-level sweep agrees
independently: its own minimum is at 81, and across 50 to 100 its cost over
the per-dataset oracle moves only from 44.0 to 43.0 percent.

**3. THE GAIN IS THE WEIGHTING SWITCH, NOT THE FAMILY SWITCH.** This is new and
it changes how the rule should be described. The rule switches two things at
one cutoff; holding one and switching the other separates them:

    policy                                              pooled error over 16 claims
    the rule: kernel + market above, lognormal + uniform below   0.2026
    kernel throughout, weighting switches at the cutoff          0.2092
    lognormal throughout, weighting switches at the cutoff       0.2096
    best fixed method (kernel, market weights)                   0.2293
    market weights throughout, family switches at the cutoff     0.2295
    uniform weights throughout, family switches at the cutoff    0.2325

**Switching only the WEIGHTING recovers three quarters of what the full rule
buys. Switching only the FAMILY recovers nothing** -- it lands on the best
fixed method, 0.2295 against 0.2293. The family switch is worth a further 3
percent on top of the weighting switch, not the other way round.

**4. The question raised about the lognormal below the cutoff is answered,
and the answer is no.** Using market weights for the lognormal below the cutoff
(`market weights throughout`) is **13.3 percent worse** than the rule and no
better than always using a kernel estimate. Section 2 says why, and the reason
is not that market share does not matter.

**Needs an author decision:** nothing blocking. Stage 3 adding the rule as a
seventh column to the scorecard figure is already decided (decision 211). The
only judgment left is how wide a range the paper prints for the cutoff.

---

## 1. The cutoff is a range, and a wide one

Thirteen cutoffs from 20 to 220, each scored on all sixteen claims against the
true parents, with the same instrument the fit-level work uses: two bootstraps
over pLCA groups, the second PAIRED so that each cutoff's excess is measured
against whichever cutoff won on that same resample.

    cutoff      20     30     40     50     60     70     81     90    100    110    130    160    220
    pooled   .2073  .2051  .2037  .2028  .2029  .2025  .2026  .2029  .2028  .2031  .2037  .2047  .2058
    in range                       yes    yes    yes    yes

The formal answer is **50 to 81**, the longest unbroken run of cutoffs whose
paired interval reaches zero. **Read it with the shape of the curve beside it**,
because the run is narrower than the flatness warrants: 90 falls out on a
bootstrap jitter of 0.0001 while 100 and 110 come back in, and the whole span
from 40 to 130 sits within 0.0011 of the best. That is the jitter decision 142
records at the fit level, and it is why the recommendation to print is a round
range rather than either endpoint.

**So what.** A practitioner who counts their EPDs and switches method somewhere
between fifty and a hundred gets essentially the best available answer. Getting
the number exactly right is worth about a fifth of what having the rule at all
is worth.

## 2. Why the rule uses uniform weights below the cutoff

The question is not whether market share matters. It is whether a
flat-Dirichlet GUESS at market share beats ignoring it when there are nine
declarations to weight. Share of datasets on which the market-weighted fit is
closer to the truth than its own uniform-weighted twin:

    declarations      3-9    10-80   81-99  100-999   1000+
    lognormal        32.8     42.5    53.6     64.6    78.7
    kernel           38.5     44.1    52.7     59.0    76.2

**It crosses half at the cutoff, for both families, and nothing was tuned to
make it do so.** Below it, guessing is worse than ignoring; above it, better.

**And knowing the shares is a different thing from guessing them**, which is
the part that makes the result make sense. From the oracle run, mean absolute
error in a material's estimated contribution:

    market shares       ignored   guessed   known
    kernel               0.1731    0.1581  0.1378
    lognormal            0.1670    0.1469  0.1267

Knowing beats ignoring on every family. What loses below the cutoff is the
flat-Dirichlet stand-in for shares nobody publishes, not weighting itself. The
rule's value is knowing **when it is worth guessing**, which is exactly why the
weighting switch is where the gain comes from.

## 3. What the rule buys, claim by claim

Every gain is against the best of the six FIXED methods on that claim -- the
comparator a reader would otherwise use -- with a paired cluster bootstrap over
pLCA groups. **All sixteen are positive and all sixteen clear zero.** Median
11.4 percent; the extremes:

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

**So what.** The claims that move most are the ones about interventions -- does
a specification cap bind, does it deliver -- which is where a design team acts.
The ones that move least are about spread and about which material drives the
uncertainty, which every method gets about equally wrong.

## 4. Where the rule is not the best choice

**It is the most accurate policy per building and not the least biased.** Its
absolute error on one building's mean total is **8.49 percent** of the true
total, the lowest of the seven, and its signed bias is **-2.86 percent**
against **+0.91** for a kernel estimate with market weights everywhere. A blend
of a family that runs low and one that runs near zero inherits a middling bias,
and bias adds across the materials of a building while noise cancels. For one
building, follow the rule; for a portfolio or a stock model, a kernel estimate
with market weights everywhere is safer.

**It captures about a fifth of what a per-material choice could buy.** Against
an oracle that picks whichever of the six is closest on each unit -- which
needs the answer in order to choose, and is a minimum over six noisy errors, so
it is optimistic -- the rule closes a median **22.5 percent** of the distance.

**The group is what limits it**, and the split carries its own control. Pooled
error by how many of a group's four materials the rule moves:

    materials moved      0      1      2      3      4
    groups             119    573    949    687    172
    the rule          .3260  .2730  .2168  .1505  .0842

At 0 and at 4 the rule IS a fixed policy and its numbers match that policy's
exactly, to 0.00e+00 on every output. Everything it buys is in the middle, and
88.4 percent of groups are in the middle.

## 5. The figure

![Choosing the method by material, claim by claim and against the cutoff](../outputs/figures/CompareUQMethods_FIG_MixedPolicy.png)

**Left:** each of the sixteen claims, the rule against the best FIXED method on
that claim, as a percentage of that method's own error, with a paired 95
percent interval over 2,500 pLCA groups. All sixteen are positive; the range is
+3.3 to +16.8 with a median of +11.4. **Right:** the same sixteen claims
pooled and each divided by its own true level, against the cutoff on a log
axis. The two dotted lines are the two fixed methods the rule interpolates
between, at 22.9 and 24.0 percent; the shaded band is the cutoffs whose paired
interval reaches zero. Across the whole sweep the cutoff moves the pooled error
by **0.47 points**, against **2.68 points** for the rule itself over the best
fixed method -- the curve is flat because the cutoff is not what matters.

*If the image does not render:* sixteen horizontal dot-and-interval rows
grouped under the five questions a reader of a probabilistic LCA asks, all to
the right of zero; beside it a shallow U-shaped curve falling from 20.7 at a
cutoff of 20 to 20.3 at 70 and rising to 20.6 at 220, far below two horizontal
reference lines at 22.9 and 24.0.

## 6. Numbers that moved

**In the study's existing tables: none.** The committed notebook was run end
to end with no error in any cell and reproduced every pre-existing table
content-identically -- and the figure byte for byte, which is also the proof
that `audits/render_figures.py` is a faithful executor of the notebook's own
bytes rather than a second author of figures. The only differing bytes
anywhere under `outputs/` are gzip header timestamps and one `written_utc`
field. That is by
construction -- the Stage 2j cells sit at the end, consume no randomness before
any existing cell, and run their own truth pass rather than extending the
study's, because a win share and a `best_method` are properties of the SET of
policies compared.

**In this stage's own first-run output:** the sweep replaced a single-cutoff
run, so every Stage 2j table is new or rebuilt. The rule's own headline numbers
are unchanged from that run to three decimals.

**One change to shared code.** `plca.swap_run` drew the same five columns six
times per pair, once per claimed saving, because only option B's use intensity
depends on the saving. The draws are now hoisted out of that loop. It is
bit-identical -- a test asserts the two routes agree exactly, and the committed
six-method design-swap table reproduces content-identically -- and it made a
23-policy sweep affordable.

## 7. Still open

| Item | |
|---|---|
| **How wide a range the paper prints for the cutoff.** Measured: best 70, formally indistinguishable 50 to 81, within 0.0011 of best from 40 to 130, fit-level optimum 81. A round "roughly 50 to 100" is supported; the wording is the author's |
| **Whether the recommendation is stated as a weighting switch with a family switch on top**, which is what item 3 of the first page measures, rather than as one two-part rule. Presentation, and it changes the emphasis of the practitioner sentence |
| **Every figure except the scorecard and this one still carries the retired weighting labels**, and the scorecard cell's own comment block still says "sampled market shares". Stage 3's caption sweep |
| **No test compares `flip.FLIP_THRESHOLDS` with the value notebook 3 recomputes.** The drift is visible only to a reader of the output. Stage 3 |
| **British spellings in files earlier stages wrote.** The deposit tidy-up |

**Settled since the last report**, and now in the decision log rather than
here: the two error definitions (decision 207 -- the paper's default is the
per-unit form everywhere, and the portfolio form only where a sentence is
explicitly about many buildings); the modality measure, which was already on
both arms and in every characteristic table; the corpus's joint
modality-and-dispersion structure, which decision 203 settled as a stated
limitation and which is not reopening generation; and the real-building anchor,
which closed with Stage 2i.

**Dropped since the last report:** the argmax qualification. The rule names the
true largest contributor 48.9 percent of the time against 52.5 for a kernel
estimate with market weights, and the study does not report that metric --
decisions 102, 143 and 155 demoted it because ranking four near-equal
contributors is not a statement a probabilistic LCA should lead with. It is in
`TABLE_MixedPolicyTruth.csv.gz` for anyone who asks.

## 8. Inputs, outputs, reproduce

**Read:** the synthetic corpus and the true parents replayed from it;
`TABLE_MethodScores.csv` and `TABLE_PLCAOracleSummary.csv` from the notebooks
above.

**Written:** thirteen tables named `TABLE_MixedPolicy*` -- the threshold curve,
the policy ranking, the per-claim gain, the fit, the weighting crossover, the
scorecard, the ceiling, the composition split and the four row-level truth
frames -- plus `CompareUQMethods_FIG_MixedPolicy.png`.

**Code:** `src/mixedpolicy.py` and `tests/test_mixedpolicy.py` (33 tests); the
hoist in `src/plca.py` and its test; nine cells at the end of notebook 3.
**625 tests pass.**

    cd notebooks && python -m nbconvert --to notebook --execute \
      --ExecutePreprocessor.kernel_name=compareuq \
      --output-dir=/tmp/nbrun --output=out.ipynb 03_CompareUQ_PerformPLCA.ipynb

About 95 minutes, of which the Stage 2j block is about 30. The figure alone
re-renders in seconds:

    python audits/render_figures.py 03_CompareUQ_PerformPLCA \
      --only "what choosing the method by material" --into-outputs

## 9. What Stage 3 picks up first

1. **Add the rule as a seventh column to the scorecard figure**, which you have
   asked for. It changes `best_method` on all sixteen rows and `stakes` on
   every row, so every sentence the paper writes about which of six methods is
   best has to be re-read against it.
2. **The caption sweep**, including the retired weighting labels everywhere
   except the two figures that use `display_method`.
