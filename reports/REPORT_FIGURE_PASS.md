# Report: the figure pass

Item 1 of section 9 of `reports/MANUSCRIPT_NARRATIVE.md`, run 2026-10-07 on the
nine main-text figures and the graphical abstract. Every result is on
`corpus_2026-09-25` at `weight_rho = 0.5`. Decision 257.

## 1. The first page

**What changed.** The nine figures are numbered `FIG1_` to `FIG9_`. Each now has
a caption under its entry in narrative section 4 that says only what is plotted.
All ten call `figstyle.apply()`. Figures 1 and 2 were built at 10 and 11.5 in
and are now 7.2 in. Figures 1, 2, 5 and 9 gained a takeaway title or a subtitle
saying what is plotted. Four label collisions or cryptic labels are fixed.
Figures 3, 7 and 9 carry new intervals, each computed in a `# TABLE` cell on
its own literal-seeded stream (decision 254). The old files are in
`archive/figures/`, so every "before" image below is the file that was
replaced.

**Second round, after the author's review the same day.** Every figure now
says "EPDs", never "declarations", and "%", never "pct" (both now rules in
`FIGURE_STYLE.md` and `reports/WRITING_STYLE.md`). Each method has one color in
every figure, from `figstyle.METHOD_COLORS`, which is exactly the palette
Figures 1 and 5 already used; Figures 7, 9 and the abstract changed color to
match. Figure 1's panels are named "probability density function (PDF)" and
"cumulative distribution function (CDF)", with more room under the subtitle.
Figure 2's panels are labeled (a) to (f), with more room under the legend.
Figure 3 has a legend of its three box styles in place of the gray text, and
its right-hand header is left-justified to its own heatmap. The two
supplement twins drawn by the Figure 3 and 4 cells picked up the same
vocabulary. **Third round:** Figure 3 is labeled as three panels, (a) the
claims heatmap with both blocks, (b) how much the answer moves, (c) the size
bands, each with a small title in the style of (c); S1, its ratio-of-means
twin, gets the same labels and its own tie boxes, from bootstrapping the ratio
of means too; Figure 7 tried standard box-plot whiskers and went back to the
10th and 90th percentiles at the author's request; the graphical abstract's headings read "Step 1:" to "Step 3:" and its
third panel shows five bars to one decimal -- normal 26.4, lognormal 18.4,
kernel estimate 17.5, size rule 17.4, size rule with market shares 13.0.
Section 2's change lists below describe the first round.

**What moved: no point estimate.** Every pre-existing table that the re-slice
rewrote came back byte-identical, except `TABLE_Building138Curves.csv.gz`, which
differs only in its gzip header and was reverted. Three tables are new.

**What needs an author decision.** Three things, in order:

1. **Approve or reject the visible changes**, shown before and after in
   section 2. Nothing is committed until you do.
2. **Figure 3's title count is fragile.** The size rule is the best of the four
   policies a reader can choose on **7 of 15** claims. On **3 of those 7** the
   runner-up is within noise: the total's mean, the total's standard
   deviation and the uncertainty index. The title still says 7, which is the
   point estimate. The figure now shows the ties, and the prose must say "clear
   of noise on 4".
3. **Figure 9's title count is fragile too.** Rebar drives the most
   uncertainty under **6 of 7** policies, but clear of Monte Carlo noise under
   only **4**: the three kernel-estimate policies and the market-weighted
   lognormal. The narrative's "effectively a tie" under the uniform-weighted
   normal is right. The same tie also holds under the market-weighted normal
   and the uniform-weighted lognormal, which the narrative does not say. The
   kernel estimates' clear rebar lead is the robust half of the finding; the
   title could say that instead. **Your wording call.**

## 2. The ten figures, before and after, with their captions

**Figure 1** `FIG1_PDFandCDFofUQMethods`. Changes: 10 in to 7.2 in; a takeaway
title and a gray subtitle replace the centered category name; the legend loses
its frame and moves from on top of the kernel estimate's far tail into the
empty lower right of panel (b); the y axes are named in the panel titles;
`figstyle.apply()`. Design unchanged.

Before: ![](../archive/figures/CompareUQMethods_FIG_PDFandCDFofUQMethods.png)
After: ![](../outputs/figures/CompareUQMethods_FIG1_PDFandCDFofUQMethods.png)

*Caption.* The six UQ methods fitted to the 204 EPDs of the EC3
category RebarSteel, each EPD divided by the category's unweighted mean
ECC. (a) Probability density function (PDF) of each fitted method; black ticks
are the EPDs, longer for a larger market weight. (b) Cumulative distribution
function (CDF) of each method, with the EPDs' empirical CDF in black.
Light shades: uniform weights; dark shades: market weights, which are drawn
because production volumes are not published.

**Figure 2** `FIG2_MetricCoverage`. Changes: 11.5 in to 7.2 in; the legend leaves
the first panel's data for a frameless row under the title; the title becomes a
message, with the coverage figure computed from the panels and rounded DOWN, so
95.9 prints as "at least 95"; panel titles shortened to "98.0 pct inside";
`figstyle.apply()`, which also turns the Unicode minus on the skewness axes into
ASCII. `TABLE_CoverageFigureStats.csv`, written by the same cell, is
byte-identical.

Before: ![](../archive/figures/CompareUQMethods_FIG_MetricCoverage.png)
After: ![](../outputs/figures/CompareUQMethods_FIG2_MetricCoverage.png)

*Caption.* Six pairs of dataset characteristics for the 10,000 synthetic
datasets (blue) and the 147 real EC3 categories (orange), characteristics under
market weights, panels (a) to (f). A ring marks a real category outside the
synthetic range on that pair, judged on a 26 by 26 grid over the pair's joint
range; each panel title gives the share of real categories inside.

**Figure 3** `FIG3_ClaimScorecard`. Changes: thin boxes for ties, plus one
subtitle line saying what a thin box means. Nothing else. The supplement's
ratio-of-means twin is byte-identical, because the intervals are computed on the
median only.

Before: ![](../archive/figures/CompareUQMethods_FIG_ClaimScorecard.png)
After: ![](../outputs/figures/CompareUQMethods_FIG3_ClaimScorecard.png)

*Caption.* (a) Median error against the true distribution, as a percentage of each
case's own true value, for fifteen claims a probabilistic LCA makes (rows,
grouped by the question a reader asks) under seven policies: three UQ families
with uniform weights and the size rule (left block), and the same three
families with market weights (right block). 2,500 synthetic pLCAs of four
materials each; the design comparison uses 2,500 design pairs. Solid box: lowest
of the left block. Dashed box: lowest of all seven, where it lies in the right
block. Thin box: a cell whose paired 95 percent bootstrap interval of difference
from its row's box reaches zero (2,000 resamples of pLCA groups or design
pairs). (b) How much the claim moves between two methods for one case, as a
percentage of that case's true value; whisker 10th to 90th percentile, box
interquartile range, line and right-hand number the median. (c) The
mean over the seven per-material claims of the same median error, by the
material's own dataset size, six fixed methods.

**Figure 4** `FIG4_MixedPolicy_SpreadZoom`. Change: the four gray "mean" labels
at the top right touched; they are now spaced one text line apart. The pixel
difference is confined to that corner.

Before: ![](../archive/figures/CompareUQMethods_FIG_MixedPolicy_SpreadZoom.png)
After: ![](../outputs/figures/CompareUQMethods_FIG4_MixedPolicy_SpreadZoom.png)

*Caption.* Error of the size rule (kernel density estimate at or above the
cutoff, three-parameter lognormal below, uniform weights; orange) and of the same
rule given the true market shares above the cutoff (blue), pooled over fifteen
claims, against the cutoff from 3 to 10,000 EPDs. Top, zoomed: solid
lines the median of per-case ratios, dashed lines the ratio of means; gray lines
the four fixed methods on each statistic. Bottom, at true scale: the size rule's
per-case error, middle half and middle 80 percent shaded; the dotted box is the
top panel's window. Vertical shading: 40 to 170 EPDs. 2,500 synthetic
pLCAs.

**Figure 5** `FIG5_WhenToUseWhich`. Change: the title named the axes ("Which
method is closest to the truth, by category size"); it now states the finding,
with a gray subtitle saying what is plotted.

Before: ![](../archive/figures/CompareUQMethods_FIG_WhenToUseWhich.png)
After: ![](../outputs/figures/CompareUQMethods_FIG5_WhenToUseWhich.png)

*Caption.* Share of synthetic datasets on which each of the six UQ methods is
closest to the market-weighted parent (W1), in a window of 800 datasets sliding
along dataset size. Gray band: 68 to 106 EPDs, the fit-level cutoffs
indistinguishable from the best. 10,000 synthetic datasets.

**Figure 6** `FIG6_BuildingDominance`. **No visible change**: the PNG is
byte-identical to the old file; only its name changed.

![](../outputs/figures/CompareUQMethods_FIG6_BuildingDominance.png)

*Caption.* The 292 North American buildings of Benke et al. (2025), A1-A3,
sorted by the ratio of the largest material's emissions to the second largest
(log axis). Shaded band: 2.18 to 2.39, the 95 percent interval of the ratio at
which the chance that the choice of UQ method changes the leading material falls
to 1 percent, calibrated on four synthetic materials; orange points lie below
its center, 2.28. Building 138 is marked.

**Figure 7** `FIG7_WeightingBySize`. Change: a shaded bar behind each median
gives its 95 percent interval, and a subtitle names the box, whiskers and bar.

Before: ![](../archive/figures/CompareUQMethods_FIG_WeightingBySize.png)
After: ![](../outputs/figures/CompareUQMethods_FIG7_WeightingBySize.png)

*Caption.* For each synthetic dataset, the W1 of a fit with uniform weights
divided by the W1 of the same family fitted with the true market shares, both
against the market-weighted parent, by dataset-size band (log2 axis; points
beyond 1/8x and 8x drawn at the limit). Kernel density estimate red,
three-parameter lognormal green. Box: interquartile range; whiskers: 10th to
90th percentile; heavy line: median; shaded bar: 95 percent bootstrap interval
of the median (2,000 resamples of datasets). Shaded band: 80 to 100
EPDs. Below: the dataset sizes of the 147 real EC3 categories, orange
below 80, with their median and interquartile range.

**Figure 8** `FIG8_ShapePlane`. Changes: "100 of 127 real categories do not"
never said what the dark points were; both classes are now named, "dark: 27 of
127 within 25 pct of the curve" and "gray: 100 further from it". The x ticks
print as 0.1, 0.3, 1, 3, 10 instead of 10^-1, 10^0, 10^1.

Before: ![](../archive/figures/CompareUQMethods_FIG_ShapePlane.png)
After: ![](../outputs/figures/CompareUQMethods_FIG8_ShapePlane.png)

*Caption.* Skewness against coefficient of variation, uniform weights, for the
127 real EC3 categories with at least ten EPDs (log x axis;
symmetric-log y axis, linear between -1 and 1). Orange curve: skewness = CV^3 +
3 CV, the only combinations a two-parameter lognormal can take. Dark points: the
27 categories whose skewness is within 25 percent of the curve; gray: the other
100.

**Figure 9** `FIG9_Building138`. Change: an interval bar on every
uncertainty-index point, and a gray subtitle with the count clear of noise.

Before: ![](../archive/figures/CompareUQMethods_FIG_Building138.png)
After: ![](../outputs/figures/CompareUQMethods_FIG9_Building138.png)

*Caption.* Building 138 of Benke et al. (2025), a multifamily residential
building in Oregon: its seven largest materials, mapped to EC3 categories, rows
in order of mean contribution. Left: each material's contribution in kgCO2e per
m2 of floor under each fitted method (solid lines and filled points, light
shades: uniform weights; dashed lines and open points, dark shades: market
weights) and under the size rule (gray fill, black diamonds). Right: the uncertainty index, each
material's share of the total's variance, under every policy, with its 95
percent interval from 1,000 bootstrap resamples of the 10,000 Monte Carlo
iterations. The other 32 materials, 12.2 percent of A1-A3, enter as a fixed
amount.

**Graphical abstract** `FIG_GraphicalAbstract`, not numbered. Change: each bar
label sat midway between two bars, so "size rule" read as belonging to either.
Each label now sits directly above its own bar.

Before: it kept its stem and was overwritten in place, so the before image is
in git: `git show 4055ccc:outputs/figures/CompareUQMethods_FIG_GraphicalAbstract.png > /tmp/ga_before.png`.
After: ![](../outputs/figures/CompareUQMethods_FIG_GraphicalAbstract.png)

*Caption.* Left: a known distribution and EPDs sampled from it. Middle: a
normal, a lognormal and a kernel density estimate fitted to those EPDs
(illustrative). Right: the pooled median error over fifteen claims for a normal,
a three-parameter lognormal and a kernel density estimate fitted with uniform
weights, the size rule, and the size rule given every product's market share,
hatched because those shares are not published.

## 2b. The proposed supplement, for review -- NOT YET CLEANED

Thirteen candidates, shown as they stand today, each with a draft caption and
the defects the supplement pass would fix. **The list is a proposal.** Nothing
here has been restyled, renumbered or checked against `FIGURE_STYLE.md` beyond
looking at it. Three were settled in the narrative already (S1 to S3); the
other ten are proposed, one per method choice or takeaway that the main text
states without showing.

**S1** `SUPP_ClaimScorecardMeanForm` -- the scorecard on the ratio of means.
Settled. It already picked up the Figure 3 vocabulary changes because one cell
draws both.

![](../outputs/figures/CompareUQMethods_SUPP_ClaimScorecardMeanForm.png)

*Draft caption.* As Figure 3, with every cell, the lower panel and the right
panel computed as total absolute error over total true value (the ratio of
means) instead of the median of per-case ratios.
*Done in the third round:* panel titles (a) to (c) and tie boxes, tested on
the ratio of means against its own row's box (3 cells tie with a best-of-four
box, 11 with a best-of-seven box, none in the size bands).

**S2** `FIG_MixedPolicy` -- the cutoff sweep on the ratio of means. Settled.

![](../outputs/figures/CompareUQMethods_FIG_MixedPolicy.png)

*Draft caption.* Error pooled over fifteen claims as the ratio of means,
against the cutoff, for the size rule (orange) and the same rule given the true
market shares above the cutoff (blue); gray lines are four fixed methods.
Shading: 40 to 170 EPDs.
*Defects:* drawn by a different cell from Figure 4, so it still says
"declarations" and "pct".

**S3** `FIG_MixedPolicy_Median` -- the same sweep on the median of ratios.
Settled.

![](../outputs/figures/CompareUQMethods_FIG_MixedPolicy_Median.png)

*Draft caption.* As S2, on the median of per-case ratios. Shading: 20 to 200
EPDs, the cutoffs indistinguishable from the best on this statistic.
*Defects:* it shades its own statistic's band, 20 to 200, where the paper
prints 40 to 170. That is correct for this figure but needs one sentence of
text.

**S4** `FIG_DemonstrateDataGeneration` -- how a synthetic dataset is built.

![](../outputs/figures/CompareUQMethods_FIG_DemonstrateDataGeneration.png)

*Draft caption.* Six synthetic datasets with one to five modes: each mode's
density (shaded, with its share of the points), the mixture parent (black),
and the drawn values (ticks). Panel titles give the mode count, the number of
values, the component overlap and the coefficient of variation.
*Defects:* in (a), (b) and (d) the x axis runs to 250 because the parent's far
tail is drawn to its truncation bound, so the whole distribution is a spike at
zero. The legends are boxed and sit on the data. The figure is 9.9 in wide.

**S5** `SUPP_DatasetExamplesByStratum` -- what the synthetic data looks like
beside the real data.

![](../outputs/figures/CompareUQMethods_SUPP_DatasetExamplesByStratum.png)

*Draft caption.* Ten synthetic datasets from each of the four size strata
(rows 1 to 4; blue, the exact parent density) and ten real EC3 categories
(bottom row; orange, a kernel density estimate, the only density available for
real data), with the values as ticks.
*Defects:* the row labels are internal codes (`s1_3_9`); the figure is 12.8 in
wide; the description sits in the title.

**S6** `SUPP_GeneratedVsEmpiricalMetrics` -- every characteristic, both arms.

![](../outputs/figures/CompareUQMethods_SUPP_GeneratedVsEmpiricalMetrics.png)

*Draft caption.* The distribution of each of 23 dataset characteristics over
the 10,000 synthetic datasets (blue) and the 147 real EC3 categories (orange),
under each weighting, with two real categories marked.
*Defects:* "(Var)" is the retired word for market weights, and "Variable"
appears in a panel title; the legend is boxed; it is not said why WoodFraming
and Gypsum are marked; the figure is 10.4 in wide.

**S7** `DEF_DemoW1Dist` -- what a Wasserstein-1 distance is.

![](../outputs/figures/CompareUQMethods_DEF_DemoW1Dist.png)

*Draft caption.* A fitted model (blue) and a small dataset (black): top, the
density and the values; bottom, the two cumulative distribution functions, with
the area between them, the Wasserstein-1 distance, shaded green.
*Defects:* the legend is boxed and sits on the curves; there are no axis
labels; it needs (a) and (b).

**S8** `FIG_BandwidthRule` -- the kernel bandwidth choice.

![](../outputs/figures/CompareUQMethods_FIG_BandwidthRule.png)

*Draft caption.* Kernel bandwidth over the data's standard deviation (top) and
W1 against the target (bottom), against dataset size, under Scott's rule,
Silverman's rule and the guarded Silverman rule the study uses, for the real
categories (left) and the synthetic datasets (right).
*Defects:* **the guard line is labeled "effective n = 30"; the shipped
threshold is 20 (decision 80)**, so this figure must be checked against the
code before it is published. It also says "variable weighting" and puts the
legend outside the panels with a frame.

**S9** `FIG_PLCATruth` -- the full error distributions behind Figure 3.

![](../outputs/figures/CompareUQMethods_FIG_PLCATruth.png)

*Draft caption.* Distribution over 2,500 synthetic pLCAs of each method's
error against the true parent, in one material's contribution (left) and in the
building total (right), where every material contributes 1.00; the orange tick
is the mean error.
*Defects:* the suptitle touches the panel titles; no panel labels.

**S10** `FIG_FlipCalibration` -- how far apart two models must be before the
answer changes.

![](../outputs/figures/CompareUQMethods_FIG_FlipCalibration.png)

*Draft caption.* Left: the chance that the top contributor changes against the
relative W1 between two fitted models, for a calibration set of two weightings
of the same data (black) and the 15 pairs of the six UQ methods (orange), with
a logistic fit (gray). Right: two densities as far apart as the 1, 5 and 10%
crossings.
*Defects:* "pct" throughout; a monospace inset; no panel labels.

**S11** `FIG_MaterialDominance` -- what a leading material buys.

![](../outputs/figures/CompareUQMethods_FIG_MaterialDominance.png)

*Draft caption.* Against the ratio of the leading material's mean contribution
to the next, in four-material synthetic pLCAs: the chance that switching UQ
method changes the leading material (top) and the change in a material's
contribution (bottom); points are pLCAs, lines rolling medians; the vertical
line is the 1% crossing.
*Defects:* the suptitle collides with the first panel title; "1 pct"; the
x-axis ticks print as 10^0.

**S12** `FIG_ChoiceDrivers` -- dataset size is enough on its own.

![](../outputs/figures/CompareUQMethods_FIG_ChoiceDrivers.png)

*Draft caption.* Share of the variation, on held-out synthetic datasets, in
which family is closer to the truth (the log ratio of the kernel estimate's W1
to the three-parameter lognormal's), explained by dataset size alone (top row,
dashed lines) and by size plus one other characteristic, under each weighting.
*Defects:* "equal weights", "market-share weights" and "(eq)" are retired
wording.

**S13** `SUPP_AllEmpiricalFits` -- every real category with its fits.

![](../outputs/figures/CompareUQMethods_SUPP_AllEmpiricalFits.png)

*Draft caption.* All 147 real EC3 categories, normalized to a mean of 1.0
(dotted line), with a histogram of the EPDs and the six fitted models; panel
titles give the category and its number of EPDs, colored by material tier.
*Defects:* large blank margins above and below the grid; at 147 panels the
curves are hard to read at print size, so this may be better split across pages.

## 3. The findings

**F1. On 3 of the 7 claims where the size rule is boxed as best, a choosable
rival is within noise.** Paired difference of the runner-up minus the rule, in
points of each case's true value, 95 percent interval over 2,000 resamples of
pLCA groups: the total's mean +0.11 [-0.06, +0.30], the total's standard
deviation +0.55 [-0.02, +1.08], the uncertainty index +0.74 [-0.26, +1.65].
The other four are clear: the total's 90th percentile +0.33 [+0.05, +0.59],
the chance of meeting a budget +0.49 [+0.37, +0.62], a material's 95th
percentile +0.29 [+0.08, +0.57], a material's share at the building's 95th
+0.67 [+0.28, +1.08]. Across all 15 rows, 12 cells sit within noise of their
row's best-of-four box and 6 within noise of the best-of-seven box. Pooled over
the fifteen claims, a kernel estimate everywhere is worse than the rule by
0.16 points [+0.01, +0.32], so the narrative's "0.2 points, clear of zero"
stands. Narrative takeaway 3 quotes 0.03 to 0.35 for this same comparison, but
that interval is at the rule's best cutoff on the threshold table, while this
one is at the code constant 80 on the main pass.
*So what:* the rule is the best a reader can choose, but on about half the
claims where it wins, a kernel estimate everywhere would do as well.

    python -c "import pandas as pd; d=pd.read_csv('outputs/tables/TABLE_ClaimScorecardIntervals.csv'); c=d[d.panel=='claims']; R='size rule\n(uniform)'; print(c[(c.best_four==R)&c.tie_four.eq(True)][['row','method','diff_lo_four','diff_hi_four']])"

**F2. In Building 138, rebar's lead in driving the uncertainty is clear only
under the kernel estimates and the market-weighted lognormal.** Lead of the
top material's uncertainty index over the runner-up, percentage points, 95
percent interval over 1,000 bootstrap resamples of the 10,000 iterations: size
rule [+10.4, +18.5], kernel estimate with uniform weights [+10.3, +18.4],
kernel estimate with market weights [+13.0, +21.0], lognormal with market
weights [+0.4, +7.2]. Within noise: lognormal with uniform weights [-2.9, +3.9],
normal with market weights [-0.1, +4.5], and normal with uniform weights, where
ready-mix 5000 psi leads [-1.7, +2.8]. Monte Carlo noise only; the fitted models
are held fixed.
*So what:* the methods that keep rebar's long right tail say clearly that rebar
is where better data would cut the uncertainty most; the methods that flatten
the tail cannot tell rebar from concrete.

    python -c "import pandas as pd; d=pd.read_csv('outputs/tables/TABLE_Building138UIInterval.csv'); print(d[d.is_leader][['method','leader','lead_lo','lead_hi','leader_clear']])"

**F3. Figure 7's medians support the 80-to-100 band.** In the 80-100 band both
medians straddle 1x: kernel estimate 1.03 [0.94, 1.12], lognormal 1.05 [0.97,
1.12]. Every band below is clear of 1x on the "hurts" side, the closest being
the lognormal at 30-79, 0.96 [0.91, 1.00], whose upper bound is 0.998. Every
band above is clear on the "helps" side, the closest being the kernel estimate
at 101-299, 1.09 [1.04, 1.14].
*So what:* below about 80 EPDs, knowing market shares measurably makes a
fit worse; above about 100 it measurably helps; in between, nobody can say.

    python -c "import pandas as pd; print(pd.read_csv('outputs/tables/TABLE_WeightingBySizeIntervals.csv')[['band','family','median_ratio','lo','hi']])"

**F4. Why the other figures get no interval.**
- Figure 1 is one illustrative category.
- Figures 2 and 8 are counts over the whole real arm, not estimates of a
  population quantity.
- Figure 4 already carries its band test (decision 224).
- Figure 5's text quotes the family crossing as an interval, 62 to 88, from the
  win-share bounds, and quotes the fit-level cutoff as the 68-to-106 band; the
  curves themselves are compared only far apart.
- Figure 6 draws the threshold as its own interval, 2.18 to 2.39, and the 292
  buildings are a census.
- The graphical abstract's bars are the pooled values F1 already bounds.

## 4. Numbers that moved

**None.** Every point estimate printed on a figure or in the narrative is
unchanged. The re-slice reproduced every existing table it rewrote
byte-identical, apart from one gzip header. The two interval cells assert their
points equal the scorecard's and the size-band table's to 1e-12, and the
Building 138 cell asserts the same for its uncertainty index. The counts in
F1 and F2 are new qualifications, not changed numbers.

    git diff --name-status 4055ccc -- outputs/tables

## 5. What is still open

- **The three qualifications in narrative section 9, item 1**, which the prose
  window must carry: F1, F2, and Figure 5's two ranges. Takeaway 8 says the
  kernel estimate "overtakes the lognormal on fit between 68 and 106
  EPDs". On the figure, 68 to 106 is the band of fit-level CUTOFFS
  indistinguishable from the best, and the families cross at 62 to 88
  (decision 142 treats these as different quantities).
- **The narrative quotes the normal's pooled error as 26.5; the table value is
  26.447, which rounds to 26.4.** A rounding slip in the quote, not a moved
  estimate; the graphical abstract prints 26.4. Decision 256 and narrative
  takeaways 1, 3 and 6 carry the 26.5.
- **Four labels in Figure 4 sit on shaded bands by design**, and
  `figstyle.check_overlaps` flags them. All four are legible at final size.
- **The supplement figures keep their stems** until the supplement list is
  decided.

## 6. Inputs and outputs

**Inputs**, all on `corpus_2026-09-25`, weight rule `rho = 0.5`: the truth-run
tables `TABLE_PLCATruth*`, `TABLE_PLCADesignSwap*`, `TABLE_MetricSizeBands.csv`,
`TABLE_MethodScores.csv` and `data/raw/building138_benke2025.csv`.

**New tables:**
- `TABLE_ClaimScorecardIntervals.csv`: notebook 3, a `# TABLE` cell before the
  scorecard figure, seed 20261007.
- `TABLE_Building138UIInterval.csv`: notebook 3, the Building 138 cell, seed
  1381, drawn after every existing draw.
- `TABLE_WeightingBySizeIntervals.csv`: notebook 2, a `# TABLE` cell before
  Figure 7, seed 20261008.

Each carries `corpus` and `weight_rho` columns.

**Figures:** `FIG1_` to `FIG9_` PNG and PDF, the redrawn
`FIG_GraphicalAbstract`, and the nine old pairs in `archive/figures/`.

**Documents:** narrative section 4 (stems and captions) and section 9 (status),
`README.md` (figure list; `FIG7_WeightingBySize` was listed under notebook 1 and
is generated in notebook 2), `CONTEXT.md` (table inventory),
`archive/README.md`, `FIGURE_STYLE.md` (stem of its worked example), and
`CLAUDE.md` decision 257.

**Tests:** all 638 pass (`python -m pytest tests/ -q`).

## 7. What the next stage picks up first

The prose, in its own window, per narrative section 9 item 2. It needs the
advisor-marked draft at an absolute path outside the repository, and it must
carry the three qualifications in section 5 above.
