# Manuscript narrative: takeaways, arc, figures

**Version 2, 2026-10-05**, after the author's review of version 1. What changed:
the eleven stale places in section 5 are **fixed** rather than listed; section 4
embeds each figure with the topic sentences it carries; six of the eleven open
questions are **settled**; and one new takeaway was added because the author's
question about imposed shape turned out to have a one-line proof.

Every number is from the tables on disk, on `corpus_2026-09-25` at
`weight_rho = 0.5`. Every reproduce command was run before this file was
written. The draft was read in place at the path the author supplied; its text
and all 97 comments were extracted to the session scratchpad under `/private/tmp`,
outside the repository tree, and nothing from it was written into the repository.

---

## 1. What the study now says, against what the draft says

| The draft's contribution | What the rework makes of it |
|---|---|
| **1. KDE fits better than normal or lognormal** | Only above about 100 declarations. Below about 50 the three-parameter lognormal is better, and the switch between them IS the recommendation. **And there is now a one-line reason why**, section 2 takeaway 2 |
| **2. W1 predicts when weighting matters and predicts pLCA differences** | Survives, narrowed twice. Whether weighting matters has a closed form in two numbers a reader already has. And a fit advantage ATTENUATES by about an order of magnitude before it reaches a pLCA claim |
| **3. UQ method selection is important and substantially affects reduction strategies** | True, and the emphasis inverts. The choice matters most for a specification cap and least for a design comparison, the normal is the one clear loser, and **most of the error is shared by all six methods and no choice removes it** |

**The one thing the draft cannot do and the rework can.** Every synthetic dataset
has a known parent, and every probabilistic LCA is run twice on the same random
draws, once with fitted models and once with the true parents. So the paper can
say how WRONG a method is, not only how different two methods are.

---

## 2. The takeaways, ranked

### 1. A probabilistic LCA is wrong by about a quarter whatever method you pick, and most of that no choice of method removes

Pooled over the fifteen claims a probabilistic LCA makes, as a percentage of each
claim's own true level: the best available choice **23.9**, the size rule
**24.0**, a kernel estimate everywhere **24.7**, a three-parameter lognormal
everywhere **24.7**, a normal everywhere **32.5**. Per claim, the part of the
error that EVERY method makes together runs from 5.9 percent of the true level
for the chance of meeting a budget to **44.6** for the uncertainty index.

**So what:** if you run a probabilistic LCA today, expect the numbers to be out
by roughly a quarter. Choosing the best available method instead of the worst
sensible one gets about one point of that back. Choosing a normal costs eight.

Decisions 174, 230, 246, 247.
`python -c "import pandas as pd; d=pd.read_csv('outputs/tables/TABLE_ClaimScorecardWithRule.csv'); print((d.pivot(index='claim',columns='method',values='total_error')*100).mean().round(2))"`

**The author's instruction, 2026-10-05: this is to be emphasized throughout.**
It is the reframing the draft most needs, and the draft inverts it.

### 2. The flexible methods win because the rigid ones cannot match spread and skew at the same time, and that is one line of algebra

For a two-parameter lognormal, **skewness = CV^3 + 3 CV**, identically. Match
the spread and the skewness is decided for you. So the two-parameter lognormal
has **no free shape parameter at all** -- the same disability as the normal's
skewness fixed at zero, not a milder version of it. The third parameter escapes
because **skewness is shift-invariant**: sigma sets the skewness and the
threshold then sets the coefficient of variation independently. A kernel
estimate constrains neither.

Measured on the real categories with ten or more declarations: **only 21.3
percent have a skewness within 25 percent of what a two-parameter lognormal of
their own spread must have.** The median category is 0.58 times as skewed as the
curve requires; 23.6 percent are more skewed. Real ECC data misses the curve in
both directions.

And the ladder predicts the study's own ordering before any fit is run. Against
the known parent, median gain over the two-parameter lognormal: at 100-999
declarations the three-parameter lognormal is **22.9 percent** closer and the
kernel estimate **33.2**; above 1,000, **25.0** and **48.1**.

**So what:** a lognormal curve has one dial for how spread out it is and that
same dial also sets how lopsided it is. Real embodied carbon data does not
oblige. Adding a third parameter, or using no curve at all, lets the model match
both at once, and that is the whole reason the flexible methods win.

Decision 252, discrepancy entries 187 and 188.
`python audits/lognormal_variants.py 1500`

**This is new, and it came from the author's question about imposed shape.** The
paper currently asserts the ordering and never explains it.

### 3. The rule a reader can actually follow is one number: count your EPDs

Uniform weights throughout, a kernel estimate at or above the cutoff, a
three-parameter lognormal below. **The cutoff is 40 to 170 declarations** and
every value in that band is indistinguishable from the best. The rule is closest
of the four options a reader can choose on **10 of 15** claims, and pooled it is
as good as knowing in advance which fixed method would win each claim (23.96
against 23.92).

**So what:** count the EPDs you have for a material. Above about a hundred, use a
kernel density estimate; below about fifty, fit a three-parameter lognormal; in
between it does not matter which.

Decisions 204, 216, 217, 220, 224, 225.

**No single-declaration cutoff is published anywhere** (decision 225).

### 4. Knowing market shares is worth four times what the rule is worth, and nobody publishes them

Giving the same rule the true market shares above the cutoff takes the pooled
error from **23.97 to 20.89 percent** of the true level: **3.08 points, or 12.8
percent of what was there.** The rule itself is worth 0.7 points.

**So what:** a probabilistic LCA is wrong by about 24 percent today. Knowing
exactly how much of each product is actually built would take that to 21. That
is four times what any choice of curve is worth.

Decisions 217, 221, 252. Entry 183.

**The author's instruction: the Discussion must go further than reporting this**
and say critically what kinds of data would most improve probabilistic LCA.
Section 3 carries the shape of that argument.

### 5. Market share does not merely fail to help below about eighty declarations; it actively hurts

Share of datasets on which the market-weighted fit is CLOSER to the truth than
its own uniform-weighted twin: **32.8 percent** at 3 to 9 declarations for the
lognormal and 38.5 for the kernel -- so at the bottom it is wrong about twice as
often as it is right -- **53.6 and 52.7** at 81 to 99, **78.7 and 76.2** above a
thousand. The Kish effective sample size has a median of **2.8** at 3 to 9
declarations, with 92.4 percent of such datasets below five effective
observations.

**So what:** with nine EPDs, two of which are the product holding ninety percent
of the market, weighting by market share rests your whole answer on two numbers.
It is aimed at exactly the right question and it is wild. You are better off
ignoring the shares until you have enough declarations inside the products that
dominate the market.

Decisions 212, 215, 219, 222, 252.
`python audits/weighting_location_shape.py --n 2000`

**Three things the paper must get right or a reviewer reads it as a modeling
error.** The crossover is in the MEAN, not the shape. It is not the kernel's
effective sample size -- the lognormal has no bandwidth and shows the same
crossover. And **the synthetic comparison is ignoring a KNOWN share against
using it, not guessing against knowing**: the weight on each product group is
that group's true share to 1.1e-16.

### 6. Do not fit a normal distribution, and say which claim you mean

A normal is **35.8 percent** worse pooled than the best available choice, and
**50.6 percent** worse on a material's chance of being the largest contributor.
It is the worst of the seven policies on 13 of 15 claims. **And on the
uncertainty index it is the best of the four a reader can choose.**

**So what:** fitting a normal curve to embodied carbon data makes your estimate
of which material matters most about half again as wrong. The one question it
does not hurt is which material drives the uncertainty, and no method answers
that one well.

Decisions 109, 148, 198.

### 7. The decision a designer makes is robust; the number they report is not

At a claimed 5 percent saving the truth is **0.599**, the six methods span
**0.609 to 0.622**, and every method is within **0.023** of the truth -- on
average over 2,500 comparisons. On ONE comparison the median absolute error runs
**0.089 to 0.140** and the six disagree about which design is better on **28
percent** of comparisons. A contribution ranking needs the leader to exceed the
next by **2.28 against 2.33** times; the one real building element available
sits at **1.02**.

**So what:** compare two designs many times and any of these methods is right on
average. Compare two designs within a few percent of each other and the method
you picked decides the answer.

Decisions 107, 118, 157, 162, 171, 172, 252.

### 8. A goodness-of-fit result overstates what the better method buys

The kernel estimate overtakes the three-parameter lognormal on fit at **81
declarations**, 68 to 106 indistinguishable. At the claim level the whole band 40
to 170 is flat and the entire sweep from 3 to 10,000 moves the pooled error by
0.78 points. The mechanism: a pLCA picks one method for all four of its
materials, so one material's advantage is averaged against three neighbors.

**So what:** a curve that fits your data better does give a better answer, but
much less of one than the fit statistic suggests.

Decisions 163, 166, 204.

**The author: "this is an important update from the previous manuscript."**

### 9. Which material drives the uncertainty is the one answer every method agrees on and every method gets wrong

NRMSE between methods **0.55**, the lowest of the main outputs, against 1.09 for
a rank-1 frequency. Recovery error against the truth **48.6 to 49.8 percent**,
the worst of the fifteen claims, and **44.6 points of that is error every method
makes together.** The cause is dataset size: a variance estimated from nine
declarations is badly understated, and a variance share has to sum to one.

**So what:** every method tells you the same thing about which material drives
your uncertainty, and all of them are about half wrong. Agreement between methods
is not evidence of accuracy.

Decisions 146, 159. Entry 186.

### 10. What the choice costs, per decision, by question

Mean over method pairs of the absolute difference in ONE decision, as a
percentage of the claim's true level: a specification cap's chance of delivering
**37.3**, how often it binds **36.4**, which material is largest **33.0**, the
uncertainty index **27.3**, down to the chance of meeting a budget **5.5**.

Decision 174, entry 186. **Every one is the error in ONE decision**; the averaged
form is 1.1 to 12.7 times smaller and must never be quoted as "the method is
right" (decision 207).

---

## 3. The narrative arc, section by section

**Introduction.** The gap it names changes: not "nobody has compared UQ methods"
but **"nobody has been able to say how wrong a UQ method is, only how different
two of them are, because nobody has had the right answer."** The pedigree
paragraph gains a measured fact: the matrix spans a geometric standard deviation
of 1.02 to 1.59 while a median real ECC category sits at 1.87, so **61.9 percent
of real categories are more dispersed than its worst possible score** (decision
194). It quantifies a different thing and the paper should say so.

**Methods.** Six methods; 147 real EC3 categories holding 116,766 declarations;
10,000 synthetic datasets stratified over four size bands whose **true parent is
known**; two evaluation levels -- fit against the known parent, and every pLCA
run a second time against the true parents on the same uniform draws. The
fifteen claims are introduced here, grouped by the five questions a reader asks.
**The shape algebra of takeaway 2 belongs here or at the head of the results.**

**Results 1: what a probabilistic LCA gets wrong, and how much of it the method
choice owns.** Takeaways 1, 6, 9, 10. The scorecard.

**Results 2: which method to use, and why.** Takeaways 2, 3, 8.

**Results 3: what not knowing market shares costs.** Takeaways 4 and 5.

**Results 4: what survives to the decision.** Takeaway 7.

**Discussion.** Six drafted limitations invert into findings; CLAUDE.md carries
that table row by row. Then the three the rework ADDS. **Then the data-collection
argument the author asked for, which the measurements now support in a specific
order:**

1. **Market-share data is worth 3.08 of the 24 points** (takeaway 4). The
   largest single lever measured, and it is not a method.
2. **It only pays above about eighty declarations** (takeaway 5), so publishing
   shares without also deepening the declaration count for the dominant products
   would not help and could hurt.
3. **Grouped shares are a far more realistic information state than full shares**
   and close part of the gap; KL2's group weight constraints are the instrument
   and the corpus can bracket the question (decision 226). Future work, stated in
   generalities.
4. **An industry-average EPD would be the one published production-weighted
   number**, and the frozen extract contains none -- all 120,280 records are
   product EPDs (decision 176). That is a concrete ask of the EPD programs.
5. **More declarations alone has a floor.** A uniform-weighted fit flattens at
   0.084 against a floor of 0.099, which is the distance between the population
   that publishes and the population that gets built. **No quantity of EPDs takes
   it below that floor** (decision 219). Only share information does.

Plus the two remedies the paper names without adopting: upper truncation
(decision 199) and the mixed policy's own limits (decision 205).

**Conclusion.** The rule, the value of market-share data, and the sentence about
agreement not being accuracy.

---

## 4. The figure selection, with the topic sentences each one carries

Seven figures. **NOT VERIFIED AT SOURCE: a web search reports that Building and
Environment caps an original research article at 10,000 words including figures
and tables but excluding references, and figures plus tables at 20 combined.**
I could not open the guide for authors to confirm either number -- both
sciencedirect.com and elsevier.com returned 403 -- and this project does not
print an unsourced figure (decision 49's withdrawn ICE number is the
precedent). **The author has institutional access and should confirm.** If the
search is right, seven figures is comfortably inside the limit and the binding
constraint is the word count rather than the figure count.

Topic sentences are written to the advisor's stated preference: a claim with a
number, first sentence of the paragraph.

### Figure 1 -- the six methods on one dataset
`FIG_PDFandCDFofUQMethods`

![](../outputs/figures/CompareUQMethods_FIG_PDFandCDFofUQMethods.png)

- "Each of the six UQ methods turns the same dataset into a different probability
  distribution, and the differences are largest in the upper tail where a carbon
  budget is written."
- "The three probability estimation methods differ in how much shape they are
  free to take: a normal distribution fixes the skewness at zero, a lognormal
  ties it to the spread, and a kernel density estimate constrains neither."

**Methods figure.** The vocabulary is correct. **Two things to settle:** it draws
a SYNTHETIC dataset (`dataset2569`), and a real EC3 category would make the
methods section concrete for the reader the author wants to reach; and the
advisor's comment 369 asks for a restructure -- "basically ePDFs and eCDFs as
uniform (on one plot) then as variable (on the other)" -- which is a different
figure from this one and would carry the weighting contrast instead of the
family contrast. **Author decides whether to rebuild it to comment 369.**

### Figure 2 -- where the real categories sit inside the synthetic cloud
`FIG_MetricCoverage`

![](../outputs/figures/CompareUQMethods_FIG_MetricCoverage.png)

- "The 10,000 synthetic datasets span the statistical characteristics of the 147
  real EC3 categories on every metric tested, leaving four uncovered
  dataset-metric pairs out of 1,470."
- "Generating datasets rather than relying on the categories EC3 happens to hold
  is what lets this study measure accuracy against a known answer, and it is why
  the findings generalize past the 147 categories available."

**This is the generalizability claim the advisor asks for three times**
(comments 282, 316, 542). The title is descriptive rather than a takeaway and
should be rewritten when the figure is finalized.

### Figure 3 -- the centerpiece
`FIG_ClaimScorecard`

![](../outputs/figures/CompareUQMethods_FIG_ClaimScorecard.png)

- "Across the fifteen claims a probabilistic LCA makes, the best method a
  practitioner can choose is still wrong by 23.9 percent of the quantity being
  claimed, and the choice between the three reasonable methods accounts for less
  than one point of that."
- "Fitting a normal distribution is the one choice that carries a real penalty,
  costing 35.8 percent of the pooled error and 50.6 percent on the question of
  which material contributes most."
- "What the choice of method costs depends entirely on which claim is being made,
  running from 5.5 percent of the true level for the chance of meeting a carbon
  budget to 37.3 percent for what a specification cap will deliver."
- "Which method is closest to the truth inverts at about a hundred declarations
  on both axes at once, from a lognormal with uniform weights below to a kernel
  estimate with market weights above."

### Figure 4 -- the rule, and what market-share data would buy
`FIG_MixedPolicy`

![](../outputs/figures/CompareUQMethods_FIG_MixedPolicy.png)

- "Switching distribution family by dataset size beats using either family
  everywhere, at any cutoff between 40 and 170 declarations."
- "Where exactly the cutoff sits is worth a fifth of what having the rule at all
  is worth: the entire sweep from 3 to 10,000 declarations moves the pooled error
  by 0.78 points, of which 0.70 is the rule beating the best fixed method."
- "Knowing the true market share of every product would reduce the error of a
  probabilistic LCA by 3.08 points of the 24 it carries, which is four times what
  any choice of fitting method is worth and is an argument for collecting
  production volumes rather than for a better curve."

### Figure 5 -- which method is closest, by category size
`FIG_WhenToUseWhich`

![](../outputs/figures/CompareUQMethods_FIG_WhenToUseWhich.png)

- "No single UQ method is closest to the truth across the range of dataset sizes
  that real ECC categories span, and which one leads changes twice."
- "The kernel estimate overtakes the three-parameter lognormal at 68 to 106
  declarations on goodness of fit, and that crossing is much less sharp once the
  fit is carried through to a probabilistic LCA claim."

**Candidate for cutting if the word count binds**, because Figure 3's lower panel
carries the same inversion at the claim level. What this adds is the FIT level,
which is what takeaway 8's attenuation is measured against.

### Figure 6 -- when a ranking claim is safe
`FIG_MaterialDominance`

![](../outputs/figures/CompareUQMethods_FIG_MaterialDominance.png)

- "A contribution ranking is only safe to report once the leading material's mean
  contribution exceeds the next by a factor of about 2.3, and the one real
  building element available in the literature sits at 1.02."
- "A dominant material protects the ranking and does nothing for the magnitude:
  the error in a material's estimated contribution is unchanged across the whole
  range of dominance."

### Figure 7 -- which categories can assume uniform weights
`FIG_WeightingDrivers`

![](../outputs/figures/CompareUQMethods_FIG_WeightingDrivers.png)

- "Whether market share matters for a material category is governed by two
  numbers a practitioner already has: how many declarations they hold and how
  spread those declarations are."
- "Only 13 of the 147 real EC3 categories are dispersed and populous enough that
  assuming uniform weights is safe, so for the great majority of materials the
  unavailability of market-share data is a live limitation rather than a
  theoretical one."

### An eighth figure worth considering, which does not exist yet

**The shape plane**: the 147 real categories plotted as coefficient of variation
against skewness, with the curve `skewness = CV^3 + 3 CV` drawn through them.
Every point on that curve is a dataset a two-parameter lognormal can represent
exactly; **21.3 percent of real categories sit within 25 percent of it**. It
makes takeaway 2 visible in one panel and would replace Figure 1's second topic
sentence. About twenty lines in a notebook cell. **Author decision whether to
build it.**

### What is cut, and why

- **`FIG_W1DistanceAndRank`, the draft's Figures 2 and 3 merged.** It scores
  every method against the variable-weighted empirical CDF of the data it was
  fitted to -- the training data, and the variable-weighted curve, so "market
  weighting improves fit" is close to true by construction. **The author has
  ruled that it does not go to the supplement as a "before" either**: there is no
  before and after, the paper is built from scratch. So it is cut outright, and
  the in-sample score is not reported.
- **`FIG_W1VsSurvivors_*` and `FIG_W1VsCharacteristic_*`, the draft's Figure 4.**
  The reduction says twenty-one marginal panels are worth about two quantities,
  and the advisor's comments 675, 766, 767 and 772 all say the figure is
  unreadable. One sentence replaces it; the twelve per-characteristic supplements
  stay available.
- **`FIG_ScatterPlot_UQResults_Subset`, the draft's Figure 5.** Method against
  method, which the run against the true parents supersedes.
- **The three-pLCA case study, the draft's Figures 6 and 7. CUT BY AUTHOR
  DECISION**, on the grounds that there is too much else to say. Two of the three
  groups were extremes selected out of 2,500 in any case. **What it was for --
  making the result concrete for a reader new to probabilistic LCA -- is now a
  writing obligation on the rest of the paper rather than a figure.**
- Everything else to the supplement.

**There is no graphical abstract in the repository.** The existing one is
`image1.png` inside the docx and I can read it: three panels, STEP 1 generate,
STEP 2 apply, STEP 3 compare. **Its third panel asserts "KDE has the best mean
fit", which is the in-sample circular result, and "Key results differ between UQ
methods", which the advisor says is hard to interpret.** Both claims are retired,
so the graphical abstract has to be rebuilt rather than relabeled.

---

## 5. The eleven stale places: fixed

All eleven are closed. Recorded as **decision 252** and discrepancy entries
**187** and **188**. What moved:

| | What was wrong | What was done |
|---|---|---|
| 1 | The safe-lead ratio was 2.13 in decisions 107 and 144; the tables say 2.28 against 2.33 at four materials | Superseding notes on both entries; current values in decision 252 |
| 2 | Every pooled figure in the log is over SIXTEEN claims; the scorecard is fifteen | Decision 252 restates them; entry 183 corrected to 23.97 / 20.89 / 3.08 points |
| 3 | The "CURRENT CANONICAL NUMBERS" block says it beats anything below it, and is pre-regeneration | Header added saying it is a dated record, not a live reference |
| 4 | The two-parameter lognormal comparison was on the superseded corpus and unstamped | **Re-run. It got stronger**: 33 to 48 percent above 100 declarations, against decision 167's 31 to 41. The script now stamps corpus and weight rule |
| 5 | Decision 218's known-share band of 50 to 100 predates the band-rule fix | Note added: it is 50 to 110; the published range is the feasible rule's 40 to 170 either way |
| 6 | Seven tables in `outputs/tables/` had **no producer anywhere in the repository** and were dated 2026-09-17 to 09-21 | Deleted, per decision 56's own precedent. `CONTEXT.md` attributed five of them to notebooks that do not write them; the inventory now names `audits/metric_reduction.py` and points at the `AUDIT_*` copies in the right directory. **No table in `outputs/tables/` now predates the regeneration** |
| 7 | `FIG_PLCATruth` printed the retired word "Variable" in its row labels | Cell now maps through `fitting.display_method`; figure redrawn in 7 seconds with the renderer |
| 8 | `TABLE_FiveStatements.csv` called the quantity reduction "method-independent" with no qualifier | Cell note corrected. **The table on disk keeps the old wording until notebook 3 next runs** -- it is a prose column, not a number, so nothing was re-run for it |
| 9 | Decision 65 forbids comparing weighting schemes cross-validated, and its exchangeability premise died with decision 190 | Recorded in decision 252 as an open question. No number moves |
| 10 | The empirical arm's headline moved and now contradicts the draft | Recorded: the lognormal is closest on 53.6 percent of the 127 categories against the kernel estimate's 26.0 |
| 11 | Decision 2 said 98 comments from named third parties | Corrected: 97, all from one author |

**The full test suite passes after every change: 632 passed in 200 seconds**,
including the notebook, renderer and figure-manifest guards that would fail on an
orphaned file or an undeclared second Generator.

    python -m pytest tests/ -q

---

## 6. Author decisions: six settled, five open

### Settled on 2026-10-05

- **The feasible rule is the recommendation.** The known-share rule reports what
  knowing market shares would buy, and ties to the recommendation that
  market-share data be collected and published.
- **The three-pLCA case study is cut.**
- **The in-sample W1 figure is cut outright**, not moved to the supplement:
  there is no before and after.
- **No specific certification credit is cited.** The paper says green building
  certifications ask for percentage reductions and that the probabilistic
  extension is a percentage reduction at a stated likelihood. No sourcing
  obligation remains.
- **The graphical abstract is rebuilt.**
- **The eleven stale places are fixed** rather than carried.

### Still open

**6.1 -- "the choice matters" or "the choice is bounded". The author is leaning
toward "the choice matters" and asked what each looks like.** Here they are as
the abstract's closing sentences, which is where the difference bites:

> **A -- the choice matters.** "Across fifteen claims scored against known
> distributions, the choice of UQ method changed what a probabilistic LCA
> reported by up to 37 percent of the quantity claimed. Fitting a normal
> distribution was 36 percent worse than the best available choice. Switching
> between a kernel density estimate and a three-parameter lognormal at a
> threshold of 40 to 170 declarations was the best available rule."

> **B -- the choice is bounded.** "Across fifteen claims scored against known
> distributions, a probabilistic LCA misstated what it reported by about 24
> percent of the quantity claimed, and the choice among reasonable UQ methods
> accounted for less than one point of that. The exceptions are specific and
> measured: fitting a normal distribution costs 36 percent, and knowing market
> shares would be worth 13 percent."

**My recommendation is A's register with B's honesty, which is neither:**

> **C.** "The choice of UQ method changed what a probabilistic LCA reported by up
> to 37 percent of the quantity claimed, and which choices matter is now
> measured rather than assumed. Fitting a normal distribution costs 36 percent of
> the error; switching family by dataset size buys 3 percent; and the largest
> lever is not a method at all but market-share data, worth 13 percent. What no
> choice of method removes is the remaining three quarters of the error."

C leads with the finding (the advisor's comment 72 asks for exactly this), keeps
the author's instruction to emphasize the shared error, and sets up the
data-collection discussion. **Author picks.**

**6.3 -- W1's role, and whether a better predictor exists.** The author asks
whether there is a metric that predicts how close your fit will be. **The study
looked and the answer is no, and that is why the size rule replaces it.** Which
method is better is driven by dataset size and essentially nothing else:
multimodality was tested and rejected (decision 88 on 127 categories, decision
139 on 10,000 out of sample), and the only large non-size effect is a quantity a
practitioner cannot compute without already knowing the market shares. So:
**W1 stays as the criterion the study scores on, and loses its billing as a
predictor.** What survives of the draft's second contribution is the
size-and-dispersion law for whether WEIGHTING matters, which is a different
question and does have a closed form. **Author confirms the demotion.**

**6.5 -- the empirical arm. The author's direction was to apply the size rule
there too, note when different methods succeed, and be careful validating
against an arm with no known parent. I ran it, and the result is honest and
slightly deflating.** Cross-validated over the 127 categories that reach n = 10:

    always a kernel estimate, uniform weights     0.3764
    always a 3-parameter lognormal, uniform       0.3462
    the size rule at a cutoff of 70               0.3463
    the size rule at a cutoff of 130              0.3449

**The rule beats always-using-a-kernel-estimate by 8 percent and is a dead heat
with always-using-a-lognormal, at every cutoff from 40 to 300.** So the empirical
arm confirms the direction and cannot separate the rule from the better fixed
method -- which is what decisions 136 and 183 already established about this arm:
127 categories cannot support the model, and the crossover is not measurable on
them. **The author's own caution is the right framing and should be in the text:
the empirical arm has no known parent, so it shows the method comparison is not
an artifact of synthetic data, and it cannot validate the rule.** Open: whether
to report this null at all, or only the characteristic comparison.

`python -c "import pandas as pd,numpy as np; d=pd.read_csv('outputs/tables/TABLE_CrossValidatedSummary.csv'); e=d[d.arm=='empirical']; w=e.pivot(index='dataset',columns='method',values='w1_cv'); n=e.groupby('dataset')['n'].first().reindex(w.index); print(round(w['KDE, Uniform'].mean(),4), round(w['Lognormal, Uniform'].mean(),4), round(np.where(n>=70,w['KDE, Uniform'],w['Lognormal, Uniform']).mean(),4))"`

**6.6 -- the three-parameter advantage. Done.** Re-run on the shipped corpus:
the three-parameter lognormal is **22.9 percent** closer to the truth than the
two-parameter at 100-999 declarations and **25.0 percent** above 1,000, and
essentially free below 10 -- which is exactly where a third parameter becomes
estimable. **No decision needed; it is a result to write.**

**6.8 -- figure count. Partly answered, and the author has to close it.** A web
search reports Building and Environment at 10,000 words including figures and
tables, excluding references, with figures plus tables capped at 20 combined --
but the guide for authors returns 403 to me on both Elsevier domains, so I have
not seen it and this file does not treat it as sourced. **The author has
institutional access.** If those numbers hold, seven figures is not the binding
constraint and the word count is. **Author confirms seven, or six by cutting
Figure 5.**

**6.11 -- citations. What I need from the author:** the **article DOI** for
Torres, Lupton, Marsh, Srubar and Allen (2026), RC&R 234, 109022 -- CLAUDE.md
carries only its Zenodo code DOI, 10.5281/zenodo.19246153 -- and **confirmation
of the Zenodo DOI for this paper's deposit**, which the draft's data statement
gives as 10.5281/zenodo.19226429. Everything else I can update from the
decision log.

---

## 7. The advisor's comments, and a style guide

All 97 comments are from Wil V. Srubar III, 2026-08-21 to 2026-08-31.

### What the rework answers structurally

| Comments | What they ask for | What answers it |
|---|---|---|
| 72, 73, 96, 97, 119 | "not ready to digest this result", "most important for what?", "be more specific" | The five-question frame: every claim named in a reader's words with a number and a true level |
| 282, 316, 542 | "stress generalizability", three times | Figure 2, and the known-parent evaluation |
| 675, 738, 759, 764, 766, 767, 772 | Figure 4 is cluttered; synthesize; order by importance; "so what" | The reduction: twenty-one panels are worth two quantities. Figure cut |
| 756 | "more important to consider X versus Y" | Takeaway 5 and Figure 7 |
| 409, 439, 464, 468, 473, 474 | "I am getting lost"; "describe these as design scenarios" | The five questions name the actions as actions |
| 829, 867, 905, 911 | "Rank #1 Frequency" is confusing | Demoted to one of five questions |
| 891, 901 | The uncertainty index paragraph repeats itself | Takeaway 9 gives the mechanism |
| 362, 378 | Too much on the goodness-of-fit tests | KS and W2 to the supplement |
| 139 | The graphical abstract claim is hard to interpret | Rebuilt |
| 765 | "harken back to Sabbie's paper" on small datasets | The size rule |

### What the rework now contradicts

**Comments 528, 541 and 592** ask for the paragraphs to focus on "KDE is best for
synthetic datasets" and "variable weighting helped all UQ methods". **Neither is
true as stated.** KDE is best above about 100 declarations and the lognormal
below; market weights help only above about 80 and hurt below. The advisor is
asking for a cleaner version of a claim that has since been narrowed, and should
be told so directly rather than have the request quietly dropped.

### Three comments that need a direct answer and are easy to miss

- **Comments 74 and 98: "I think I edited this to be (a) correct and (b) a bit
  more clear - yes?"** Questions TO the author about edits the advisor made, on
  two abstract sentences. Both sentences are retired, so the answer is neither
  yes nor no and has to be given.
- **Comment 491: "why? because the materials that contribute the most to variance
  are focused on"** -- on the uncertainty index. Takeaway 9 is the answer and has
  never been written for a reader.

### The style guide

**The author asked whether to build one from the advisor's comments. Yes, and it
is cheap, because the comments are unusually consistent about four things:**

1. **A strong topic sentence carrying the claim, first.** Comments 528, 541, 592,
   645, 762, 766 all say a version of this. It is the single most repeated note.
2. **One idea per paragraph.** Comments 592, 645 ("at least 3 I can count"), 891.
3. **Teach the reader before using a term.** Comments 829 ("the reader does not
   yet know what Rank #1 Frequency"), 96, 97, 72.
4. **Say "so what" or cut it.** Comments 738, 759, 764, 767 -- "a bit thin",
   "even needed?"

**What I would like from the author**, in order of usefulness:

- **The existing style guide the previous Claude agent produced.** If it already
  covers the four above, we extend it rather than start over.
- **One or two of the advisor's own publications** would help calibrate register
  and paragraph length, which comments cannot show.

The guide would live at `reports/WRITING_STYLE.md` and be binding on the draft
the same way `FIGURE_STYLE.md` is binding on figures.

---

## 8. How to proceed

**This window should not draft prose.** Three things remain before drafting, and
they are cheap:

1. **The author answers 6.1, 6.3, 6.5, 6.8 and 6.11**, and sends the existing
   style guide and the KL2 DOI.
2. **One window writes `reports/WRITING_STYLE.md`** from the 97 comments plus
   whatever the author sends, and decides the graphical abstract and whether to
   build the shape-plane figure.
3. **Then a drafting window per section**, working from the topic sentences in
   section 4 and the arc in section 3, with this file and the decision log.

**Stay in Claude Code rather than moving to the web.** The reference PDFs are in
`refs/` here and nowhere else, the tables behind every number are here, and the
draft is readable in place from here. A web window would be working from memory
on exactly the numbers this file has just spent a session correcting.

**One caution for whoever drafts.** Section 5 of this file is a record of eleven
places where the decision log disagreed with the tables. **Quote the tables.**
The log is the history of how a number was arrived at, not the number.

### The repository state, which is UNCOMMITTED

Nothing in this session was committed, because the author did not ask for it.
The working tree carries, on branch `stage-4-deposit`:

    M  CLAUDE.md                      superseding notes on 107, 144, 218, 2;
                                      decision 252 appended
    M  CONTEXT.md                     table inventory corrected (6 rows)
    M  reports/MANUSCRIPT_discrepancies.md
                                      canonical block flagged; entry 183
                                      corrected; entries 187 and 188 added
    M  audits/lognormal_variants.py   provenance stamping
    M  notebooks/03_...ipynb          cells 86 and 94
    M  outputs/figures/..._FIG_PLCATruth.{png,pdf}   redrawn
    D  7 orphan tables under outputs/tables/
    ?? reports/MANUSCRIPT_NARRATIVE.md             this file

It also carries changes made BEFORE this session, from the 2026-10-05 scorecard
review that produced discrepancy entry 186: `src/metricset.py`,
`tests/test_metricset.py`, `reports/START_HERE.md`,
`outputs/tables/TABLE_ClaimChoiceCost.csv`,
`reports/REVIEW_scorecard_uncertainty_index.md` and the scorecard figure.
**Those are not mine and should be committed separately if the author wants the
history to bisect cleanly.**

**The full suite passes on the tree as it stands: 632 tests.**
