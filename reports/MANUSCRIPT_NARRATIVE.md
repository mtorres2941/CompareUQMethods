# Manuscript narrative

What the paper says, in what order, on which figures. Written to be read as the
plan itself rather than as a record of how it was arrived at; the history is in
the decision log and in `reports/MANUSCRIPT_discrepancies.md`.

Every number is from the tables on disk, on `corpus_2026-09-25` at
`weight_rho = 0.5`. **Building and Environment allows 10,000 words excluding
references, 15 figures and 5 tables**, so eight figures and one table is less
than half the figure allowance and the binding constraint is the word count.

`reports/WRITING_STYLE.md` is binding on the prose.

---

## 1. The thesis

Nobody has been able to say how *wrong* a UQ method is for building material
emissions -- only how *different* two methods are -- because nobody has had the
right answer to compare against. This study manufactures one: 10,000 synthetic
ECC datasets whose true distribution is known by construction, and a
probabilistic LCA run twice on the same random draws, once with fitted models
and once with the true distributions.

That lets the paper say three things the literature cannot. **A probabilistic
LCA misstates what it reports by about a quarter of the quantity being claimed,
and most of that no choice of method removes.** **Two choices do matter and are
measured: never fit a normal distribution, which costs 36 percent of the error,
and switch distribution family by dataset size, which buys 3 percent.** **And
the largest lever is not a method at all -- it is market-share data, worth 13
percent.**

---

## 2. The takeaways, ranked

### 1. A probabilistic LCA is wrong by about a quarter whatever method you pick, and most of that is shared

Pooled over the fifteen claims a probabilistic LCA makes, as a percentage of
each claim's own true level: the best available choice **23.9**, the size rule
**24.0**, a kernel estimate everywhere **24.7**, a three-parameter lognormal
everywhere **24.7**, a normal everywhere **32.5**. Per claim, the part of the
error that *every* method makes together runs from 5.9 percent of the true level
for the chance of meeting a budget to **44.6** for the uncertainty index.

**So what:** expect the numbers a probabilistic LCA gives you to be out by
roughly a quarter. Choosing the best available method instead of the worst
sensible one gets about one point of that back; choosing a normal costs eight.

Decisions 174, 230, 246, 247.
`python -c "import pandas as pd; d=pd.read_csv('outputs/tables/TABLE_ClaimScorecardWithRule.csv'); print((d.pivot(index='claim',columns='method',values='total_error')*100).mean().round(2))"`

### 2. The flexible methods win because the rigid ones cannot match spread and skew at once

For a two-parameter lognormal, **skewness = CV^3 + 3 CV**, identically. Match
the spread and the skewness is decided for you, so the family has **no free
shape parameter at all** -- the same disability as the normal's skewness fixed
at zero, not a milder version of it. The third parameter escapes because
skewness is shift invariant: sigma sets the skewness and the threshold then sets
the coefficient of variation independently. A kernel estimate constrains
neither.

Only **21.3 percent** of real categories with ten or more declarations have a
skewness within 25 percent of what a two-parameter lognormal of their spread
requires; the median is **0.58** times as skewed as the curve demands and 23.6
percent are more skewed, so the family is the wrong shape in both directions.

And the ladder predicts the study's own ordering. Against the known parent,
median gain over the two-parameter lognormal: at 100-999 declarations the
three-parameter lognormal is **22.9 percent** closer and the kernel estimate
**33.2**; above 1,000, **25.0** and **48.1**.

**So what:** a lognormal curve has one dial for how spread out it is, and that
same dial sets how lopsided it is. Real embodied carbon data does not oblige.

Decision 252, entries 187, 188, 189. `python audits/lognormal_variants.py 1500`

**This is what the field actually uses, and that is now sourced.** ecoinvent's
default is a two-parameter lognormal -- "the geometric mean and the geometric
standard deviation" (Muller et al. 2016, IJLCA 21:1327-1337) -- and the same
authors name "the imposition of the lognormal" as the first of three
limitations of the pedigree approach, and say that other distributions "are more
appropriate when they better represent the uncertainty associated with the
datum. Most often, this will be the case when the basic uncertainty has been
calculated based on available data." **The default is for the no-data case. This
paper is about the case where you have the data.**

### 3. The rule a reader can follow is one number: count your EPDs

Uniform weights throughout, a kernel estimate at or above the cutoff, a
three-parameter lognormal below. **The cutoff is 40 to 170 declarations**, every
value in that band indistinguishable from the best. The rule is closest of the
four options a reader can choose on **10 of 15** claims, and pooled it is as
good as knowing in advance which fixed method would win each claim (23.96
against 23.92).

**So what:** count the EPDs you have. Above about a hundred use a kernel density
estimate, below about fifty fit a three-parameter lognormal, in between it does
not matter which.

Decisions 204, 216, 217, 220, 224, 225. **No single-declaration cutoff is
printed anywhere in the paper.**

### 4. Knowing market shares is worth four times what the rule is worth

Giving the same rule the true market shares above the cutoff takes the pooled
error from **23.97 to 20.89 percent** of the true level: **3.08 points, or 12.8
percent of what was there**. The rule itself is worth 0.7 points.

**So what:** a probabilistic LCA is wrong by about 24 percent today. Knowing
exactly how much of each product is actually built would take that to 21 -- four
times what any choice of curve is worth.

Decisions 217, 221, 252. Entry 183.

### 5. Market share does not merely fail to help below about eighty declarations; it actively hurts

Share of datasets on which the market-weighted fit is *closer* to the truth than
its own uniform-weighted twin: **32.8 percent** at 3 to 9 declarations for the
lognormal and 38.5 for the kernel -- wrong about twice as often as right --
**53.6 and 52.7** at 81 to 99, and **78.7 and 76.2** above a thousand. Under
true weights the Kish effective sample size has a median of **2.8** at 3 to 9
declarations, with 92.4 percent of such datasets below five effective
observations.

**So what:** with nine EPDs of which two are the product holding ninety percent
of the market, weighting by market share rests your whole answer on two numbers.
It is aimed at exactly the right question and it is wild. Better to ignore the
shares until enough declarations sit inside the products that dominate the
market.

Decisions 212, 215, 219, 222, 252.

**Two things the paper must say or a reviewer reads this as a modeling error.**
The effect is in the estimate of the **mean**, not in the distribution's shape,
so it is not about kernels or bandwidths -- the three-parameter lognormal has no
bandwidth and shows the same crossover. And **the synthetic comparison is
ignoring a known share against using it, not guessing against knowing**: the
weight on each product group is that group's true share to 1.1e-16.

### 6. Do not fit a normal distribution, and say which claim you mean

A normal is **35.8 percent** worse pooled than the best available choice, and
**50.6 percent** worse on a material's chance of being the largest contributor.
It is the worst of the seven policies on 13 of 15 claims. **And on the
uncertainty index it is the best of the four a reader can choose.**

**So what:** fitting a normal curve to embodied carbon data increases the error
in your estimate of which material contributes most by about half. The one
question it does not hurt is which material drives the uncertainty, and no
method answers that one well.

Decisions 109, 148, 198.

### 7. The decision a designer makes is robust; the number they report is not

At a claimed 5 percent saving the truth is **0.599**, the six methods span
**0.609 to 0.622**, and every method is within **0.023** of the truth -- on
average over 2,500 comparisons. On *one* comparison the median absolute error
runs **0.089 to 0.140** and the six disagree about which design is better on
**28 percent** of comparisons. A contribution ranking needs the leader to exceed
the next by **2.28 against 2.33** times; the one real building element available
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

Decisions 163, 166, 204.

### 9. Which material drives the uncertainty is the one answer every method agrees on and every method gets wrong

NRMSE between methods **0.55**, the lowest of the main outputs, against 1.09 for
a rank-1 frequency. Recovery error against the truth **48.6 to 49.8 percent**,
the worst of the fifteen claims, and **44.6 points of that is error every method
makes together**. The cause is dataset size: a variance estimated from nine
declarations is badly understated, and a variance share must sum to one.

**So what:** every method tells you the same thing about which material drives
your uncertainty, and all of them are about half wrong. **Agreement between
methods is not evidence of accuracy** -- which is the general lesson of the
paper.

Decisions 146, 159. Entry 186.

### 10. What the choice costs, per decision, by question

Mean over method pairs of the absolute difference in one decision, as a
percentage of the claim's true level: a specification cap's chance of delivering
**37.3**, how often it binds **36.4**, which material is largest **33.0**, the
uncertainty index **27.3**, down to the chance of meeting a budget **5.5**.

Decision 174, entry 186. Every one is the error in **one** decision; the averaged
form is 1.1 to 12.7 times smaller and must never be quoted as "the method is
right".

### 11. A better fit does give a better answer, and not by as much as the fit suggests

Within a material, ranking the six methods by how well they fit and by how wrong
their answer is gives a median rank correlation of **+0.83**, positive on **88.2
percent** of the 10,000 materials; the best-fitting method is also the most
claim-accurate **42.8 percent** of the time against a 16.7 percent chance level.
What does not transfer is the magnitude.

**So what:** there is no check you can run on your own data to find out whether
your choice of method will matter for your building. Follow the rule precisely
because you cannot tell.

This is the summarizing takeaway and where the draft's retired second
contribution lands. **It goes last in the Results and feeds the Conclusion.**
Recomputed on the shipped corpus; decision 166's +0.600, 82.2 and 39.0 are from
a superseded one.

---

## 3. The arc, section by section

**Introduction.** The gap: nobody has been able to say how wrong a UQ method is,
only how different two are. The pedigree paragraph says what the field defaults
to -- a two-parameter lognormal, cited to both Muller et al. (2016) papers --
quotes their own first limitation, and notes that their guidance points away
from the default precisely when data is available, which is this paper's case.
The matrix also spans a geometric standard deviation of 1.02 to 1.59 while a
median real ECC category sits at 1.87, so **61.9 percent of real categories are
more dispersed than its worst possible score**; it quantifies a different thing
and the paper says so.

**Methods.** Six methods; 147 real EC3 categories holding 116,766 declarations;
10,000 synthetic datasets stratified over four size bands whose **true parent is
known**; two evaluation levels -- fit against the known parent, and every pLCA
run a second time against the true parents on the same uniform draws. The
fifteen claims are introduced here, grouped by the five questions a reader asks.
**Takeaway 2's shape algebra belongs here or at the head of the Results.**

**Results 1: what a probabilistic LCA gets wrong, and how much the method choice
owns.** Takeaways 1, 6, 9, 10.

**Results 2: which method to use, and why.** Takeaways 2, 3, 8.

**Results 3: what not knowing market shares costs.** Takeaways 4 and 5.

**Results 4: what survives to the decision.** Takeaway 7, then takeaway 11.

**Discussion.** Six drafted limitations invert into findings; CLAUDE.md carries
that table row by row. Then the three the rework adds. **Then the
data-collection argument, which the measurements support in this order:**

1. Market-share data is worth **3.08 of the 24 points** -- the largest single
   lever measured, and not a method.
2. It only pays above about eighty declarations, so publishing shares without
   also deepening the declaration count for the dominant products would not help
   and could hurt.
3. **Of the 134 real categories where market shares matter, about two thirds are
   too small to use an estimate of them** (section 4, Figure 7). For those the
   uncertainty is irreducible by modeling.
4. Grouped shares are a far more realistic information state than full shares and
   close part of the gap; KL2's group weight constraints are the instrument
   (decision 226). Future work, in generalities.
5. An industry-average EPD would be the one published production-weighted number,
   and the frozen extract contains none -- all 120,280 records are product EPDs
   (decision 176). A concrete ask of the EPD programs.
6. **More declarations alone has a floor.** A uniform-weighted fit flattens at
   0.084 against a floor of 0.099, the distance between the population that
   publishes and the population that gets built. No quantity of EPDs takes it
   below that floor (decision 219). Only share information does.

Plus the remedies the paper names without adopting: upper truncation (decision
199) and the rule's own limits (decision 205).

**Conclusion.** The rule, the value of market-share data, and the sentence about
agreement not being accuracy.

---

## 4. The figures

Eight figures. Topic sentences are written to `WRITING_STYLE.md` principle 1: a
claim with a number, first sentence of the paragraph.

### Figure 1 -- the six methods on one real category
`FIG_PDFandCDFofUQMethods`

![](../outputs/figures/CompareUQMethods_FIG_PDFandCDFofUQMethods.png)

- "Each of the six UQ methods turns the same 204 declarations of reinforcing
  steel into a different probability distribution, and they disagree most about
  the upper tail, where a carbon budget is written."
- "The three probability estimation methods differ in how much shape they are
  free to take: a normal distribution fixes the skewness at zero, a lognormal
  ties it to the spread, and a kernel density estimate constrains neither --
  which is why only the kernel estimate reproduces the second group of
  declarations near twice the mean."

Reinforcing steel rather than a synthetic dataset: a reader cannot picture
`dataset2569`, every structural engineer names rebar, it holds 204 declarations
so the kernel estimate is what the paper recommends for it, and its second hump
near 2.1 makes the families visibly disagree. It is also the thread for the
worked example in section 5.

### Figure 2 -- where the real categories sit inside the synthetic cloud
`FIG_MetricCoverage`

![](../outputs/figures/CompareUQMethods_FIG_MetricCoverage.png)

- "The 10,000 synthetic datasets span the statistical characteristics of the 147
  real EC3 categories on every characteristic tested, leaving four uncovered
  dataset-metric pairs out of 1,470."
- "Generating datasets rather than relying on the categories EC3 happens to hold
  is what lets this study measure accuracy against a known answer, and it is why
  the findings generalize past the 147 categories available."

**The caption must say why these six pairs.** They are the leading
characteristics of the principal components of the full characteristic set,
paired so each panel shows two that are *not* redundant with one another, plus
two pairs of direct interest: dataset size against the effect of weighting, and
the two goodness-of-fit statistics against each other. The coefficient of
variation appears in two panels because it loads on two different components.
Without that sentence the pairing looks arbitrary.

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
- "How much the answer moves when you pick a different method depends entirely on
  which claim is being made, and on most claims the spread across buildings is
  wider than its own average."
- "Which method is closest to the truth inverts at about a hundred declarations
  on both axes at once, from a lognormal with uniform weights below to a kernel
  estimate with market weights above."

The right-hand bar now spans the **10th to 90th percentile of the per-building,
per-method-pair difference** with the median marked, rather than a single
averaged number: a claim where every building moves a little and one where most
move nothing and a few move a lot have the same mean and different spreads, and
only the second is a reason to worry about one building.

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
  any choice of fitting method is worth."

### Figure 5 -- which method is closest, by category size
`FIG_WhenToUseWhich`

![](../outputs/figures/CompareUQMethods_FIG_WhenToUseWhich.png)

- "No single UQ method is closest to the truth across the range of dataset sizes
  real ECC categories span, and which one leads changes twice."
- "The kernel estimate overtakes the three-parameter lognormal at 68 to 106
  declarations on goodness of fit, and that crossing is much less sharp once the
  fit is carried through to a probabilistic LCA claim."

First candidate for cutting if the word count binds: Figure 3's lower panel
carries the same inversion at the claim level.

### Figure 6 -- when a ranking claim is safe
`FIG_MaterialDominance`

![](../outputs/figures/CompareUQMethods_FIG_MaterialDominance.png)

- "A contribution ranking is only safe to report once the leading material's mean
  contribution exceeds the next by a factor of about 2.3, and the one real
  building element available in the literature sits at 1.02."
- "A dominant material protects the ranking and does nothing for the magnitude:
  the error in a material's estimated contribution is unchanged across the whole
  range of dominance."

Cleaned up -- no shouting capitals, short axis labels, and the two faint notes
about the deliberate 10:1 and 100:1 test cases moved to the printout. **It is
still a monotone decline beside a flat line, and both are shapes a sentence
states exactly as well.** My recommendation is to cut it and keep the two
findings as sentences; it is embedded here so the judgment can be made on the
cleaned-up version.

### Figure 7 -- where not knowing market shares actually bites
`FIG_WeightingDrivers`

![](../outputs/figures/CompareUQMethods_FIG_WeightingDrivers.png)

- "For 134 of the 147 real EC3 categories, swapping uniform weights for a
  plausible market share moves the fitted distribution by more than the amount
  that changes which material leads five percent of the time."
- "About two thirds of those categories hold too few declarations for a
  market-share estimate to improve the fit, so for them the uncertainty that
  missing market-share data creates cannot be removed by any modeling choice --
  only by more declarations in the products that dominate the market, or by
  published shares."

Rebuilt. The previous version asked which categories can "safely" assume uniform
weights, and that word invited the reader to conclude "unsafe here, so weight
here" -- the opposite of the recommendation. The two axes carry the two
different things being measured, and the counts are in the quadrants.

**THE TWO CRITERIA, EXACTLY, because the caption has to carry them and because
they sound contradictory until they are stated.**

**"Market shares matter"** is a statement about SENSITIVITY, measured on the
real categories. For each one, draw 1,000 market-share vectors from the coherent
block model at `rho = 0.5`, fit the distribution under each, and take the MEDIAN
W1 distance between that fit and the uniform-weighted fit, in units of the
dataset's own mean. A category counts when that median exceeds **0.015**, which
is `flip.FLIP_THRESHOLDS[0.05]` -- the model distance at which the top
contributor of a four-material pLCA changes 5 percent of the time. **134 of
147.**

**"Too small to use them"** is a statement about ACCURACY, and its threshold is
measured on the SYNTHETIC arm because only there is there a known truth to be
closer to. The market-weighted fit beats its own uniform twin on 42 to 44
percent of datasets below 80 declarations and 53 percent above, so 80 is where
using a share starts to pay. **93 of the 134, which is 69 percent.**

**Three qualifications the paper owes.** The two criteria come from different
arms, and combining them assumes the synthetic crossover transfers to real
categories -- which cannot be checked, because real categories have no known
parent. The sensitivity count moves with the flip level: 144 at 1 percent, 134
at 5, 117 at 10. And the 69 percent moves with the size cut: 48 percent at
n < 40, 69 at n < 80, 83 at n < 170. **So the paper says "about two thirds" and
not "93 of 134".**

### Figure 8 -- why the rigid families lose
`FIG_ShapePlane`

![](../outputs/figures/CompareUQMethods_FIG_ShapePlane.png)

- "A two-parameter lognormal has no freedom to choose its shape: once its spread
  is matched to the data its skewness is fixed at CV^3 + 3 CV, so every dataset
  it can represent exactly lies on a single curve."
- "Only 27 of 127 real ECC categories sit within 25 percent of that curve, and
  the median category is 0.58 times as skewed as a lognormal of its spread
  requires, so the family ecoinvent defaults to is systematically the wrong shape
  for this data."

If it is cut, the algebra and the 21.3 percent survive as two sentences and lose
little: the equation is the finding and the figure illustrates it.

### Cut

**`FIG_W1DistanceAndRank`** (the draft's Figures 2 and 3) scores every method
against the variable-weighted empirical CDF *of the data it was fitted to*, so
"market weighting improves fit" is close to true by construction. Cut outright,
not to the supplement.

![](../outputs/figures/CompareUQMethods_FIG_W1DistanceAndRank.png)

**`FIG_W1VsSurvivors_*`** (the draft's Figure 4) is twenty-one marginal panels
worth about two quantities, and is in-sample.

![](../outputs/figures/CompareUQMethods_FIG_W1VsSurvivors_Synthetic.png)

**`FIG_ScatterPlot_UQResults_Subset`** (the draft's Figure 5) is method against
method, which the run against the true parents supersedes: it can show that two
methods disagree and never which is right.

![](../outputs/figures/CompareUQMethods_FIG_ScatterPlot_UQResults_Subset.png)

**The three-pLCA case study** (the draft's Figures 6 and 7): two of its three
groups are extremes selected out of 2,500.

![](../outputs/figures/CompareUQMethods_FIG_PLCAVisualizeUQFits.png)

Everything else to the supplement.

---

## 5. The worked example

The paper is abstract from end to end -- 10,000 synthetic datasets, every
material normalized to a mean of 1.0. A reader who designs buildings has nothing
concrete to hold, and the advisor's covering instruction is to teach.

**One real four-material building runs through the paper, and it never carries
an accuracy claim.** `ReadyMix [4000-4999 psi]` at 31,025 declarations, `Gypsum`
at 771, `RebarSteel` at 204, `BlanketInsulation [mineral wool]` at 184 -- a
designer names all four. It is introduced in the Methods as the thing Figure 1
draws and referred back to once in each Results subsection: what the six methods
say its total is, which material each names as largest, what a specification cap
would deliver.

**Every accuracy claim stays on the synthetic arm, because a real category has
no known parent**, and the text says so once. That is the one line that separates
a worked example from a case study, and it is why the three-pLCA case study was
cut rather than rebuilt: it was being asked to carry evidence it could not carry.

It costs about 250 words and no separate figure. **One gap: all four materials
sit above the cutoff, so the size rule does not bite on this building.** Either
add a fourth material below 40 declarations, or use the building for the claims
and make the rule's bite a separate sentence.

---

## 6. The graphical abstract

The existing one is three panels: generate 10,000 datasets, apply 6 UQ methods,
compare goodness-of-fit and pLCA results. Its third panel asserts "KDE has the
best mean fit", which is the in-sample circular result the paper no longer
reports, and "Key results differ between UQ methods", which is no longer the
headline. Panels one and two need units and a legend.

**Proposed, same three-panel structure:**

| | Proposed |
|---|---|
| **1** | **"We generate 10,000 datasets whose true distribution we know"** -- one dataset drawn from a parent with the parent curve behind it, x axis labeled ECC / mean ECC. The known truth is the paper's whole method and panel 1 should say so |
| **2** | The six fits, with "uniform weights" and "market weights" replacing the retired labels |
| **3** | **"A probabilistic LCA is wrong by about a quarter, and here is what you can control"** -- one stacked bar: the error the best available method still makes, plus what the choice of method adds, plus what market-share data would buy |

Panel 3 is the paper in one picture and is a simplification of Figure 3's logic
rather than new analysis. A PowerPoint rebuild; it does not touch the repository.

---

## 7. What is still open

1. **The worked example's fourth material** -- all four candidates sit above the
   cutoff (section 5).
2. **Figure 6: cut, as recommended, or keep the cleaned-up version?**
3. **Figure 8: keep?**
4. **Figure 5: keep at eight figures, or cut to seven?**
5. **The KL2 code DOI** ends 154 in the author's citation and 153 in the project
   brief -- probably Zenodo's concept-versus-version pair, worth one check.

---

## 8. Reference material

- **Quote the tables, not the decision log.** Eleven places where the two
  disagreed were corrected on 2026-10-05 and the log was the stale side in every
  one (decision 252).
- **`reports/WRITING_STYLE.md`** is binding on the prose: nine principles, the
  voice convention, the controlled vocabulary, the rules for numbers, and a test
  to run before any section is handed over.
- **`reports/MANUSCRIPT_discrepancies.md`** carries every place the draft and the
  code disagree, 189 entries.
- **The advisor stopped editing partway through** and said he wants to see how
  this version responds before reading more. The unreviewed final third is the
  real test, because it is the part no reviewer has already repaired.
