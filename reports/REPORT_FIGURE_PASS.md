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

## 2b. The supplement: twelve figures, numbered and restyled

Numbered in the order the manuscript first needs them, methods before results.
All twelve call `figstyle.apply()`, say "EPDs", use the % symbol and the
settled weighting words, carry panel labels where they have panels, and have a
takeaway title with a gray subtitle saying what is plotted. The old unnumbered
files are in `archive/figures/`; each "before" image below is that file.

**SUPP1** `SUPP1_DemonstrateDataGeneration`. **The flat panels were a drawing
defect, not a generation one.** Each curve was drawn on a 1,000-point grid
running to the parent's truncation bound, up to 240 times the mean, a step of
0.24, which is coarser than the whole body of the distribution. The values
drawn in those panels top out at 3 to 5 times the mean. The grid now ends at
the larger of the parent's 99.9th percentile and the largest value drawn. A
second defect surfaced once the shapes could be seen: each mode was drawn
untruncated while the mixture is truncated, so the modes fell short of the
black curve; they are now truncated the same way, and the cell asserts they
sum to it. The drawn values are byte-identical, so no random number moved.

Before: ![](../archive/figures/CompareUQMethods_FIG_DemonstrateDataGeneration.png)
After: ![](../outputs/figures/CompareUQMethods_SUPP1_DemonstrateDataGeneration.png)

*Caption.* Six synthetic datasets with one to five modes, (a) to (f): each
mode's density scaled by its share of the values (shaded), the mixture parent
they are drawn from (black), and the values drawn (ticks). Panel titles give the
number of modes and values, the mean pairwise component overlap and the
coefficient of variation.

**SUPP2** `SUPP2_DatasetExamplesByStratum`. Transposed to fit the page: a
column per size band, plus the real categories, and a row per example, 7.2 in
wide instead of 16. Column headers replace the internal codes.

Before: ![](../archive/figures/CompareUQMethods_SUPP_DatasetExamplesByStratum.png)
After: ![](../outputs/figures/CompareUQMethods_SUPP2_DatasetExamplesByStratum.png)

*Caption.* Ten synthetic datasets from each of four size bands (blue, the exact
parent density each was drawn from) and ten real EC3 categories (orange, a
kernel density estimate of the EPDs), with the values as ticks. Panel titles
give the number of values, the coefficient of variation and, for synthetic
datasets, the number of modes.

**SUPP3** `SUPP3_GeneratedVsEmpiricalMetrics`. Each characteristic's two
weightings now sit side by side, so the panel titles fit; the retired "(Var)"
and "Uniform vs Variable" are gone from `src/dct_metriclabels.json`, which
every figure reading characteristic labels shares; the two marked categories
are named as references in the legend.

Before: ![](../archive/figures/CompareUQMethods_SUPP_GeneratedVsEmpiricalMetrics.png)
After: ![](../outputs/figures/CompareUQMethods_SUPP3_GeneratedVsEmpiricalMetrics.png)

*Caption.* The distribution of each dataset characteristic over the 10,000
synthetic datasets (blue) and the 147 real EC3 categories (orange, with a tick
per category), under market and under uniform weights. Fit
statistics are Shapiro-Francia; the fitted modality index uses the bandwidth the
study fits. Two real categories are marked for reference. Wide-ranging
characteristics are on log or symmetric-log axes.

**SUPP4** `SUPP4_DemoW1Dist`. The framed legend that sat on the CDF is
replaced by labels on the curves; panels labeled; a title.

Before: ![](../archive/figures/CompareUQMethods_DEF_DemoW1Dist.png)
After: ![](../outputs/figures/CompareUQMethods_SUPP4_DemoW1Dist.png)

*Caption.* A fitted model (blue) and a small dataset (black). (a) The model's
probability density and the data as ticks, tick length showing weight. (b) The
two cumulative distribution functions; the shaded area between them is the
Wasserstein-1 distance (W1).

**SUPP5** `SUPP5_BandwidthRule`. **The guard line was drawn at 30 and labeled
30; the shipped threshold has been 20 since decision 80.** It now reads
`customstats.SILVERMAN_MIN_NEFF`. The W1 rows were labeled "against the
target"; they are in-sample, each fit scored against the data it was fitted
to, and now say so.

Before: ![](../archive/figures/CompareUQMethods_FIG_BandwidthRule.png)
After: ![](../outputs/figures/CompareUQMethods_SUPP5_BandwidthRule.png)

*Caption.* Kernel bandwidth over the data's standard deviation (a, b) and
in-sample W1 (c, d) against dataset size under Scott's rule (red), Silverman's
rule (blue dashed) and the guarded Silverman rule the study uses (black), for
the real EC3 categories (a, c) and a sample of synthetic datasets (b, d), all
under market weights. Lines are rolling means; points are datasets. The dotted
line marks the guard's threshold of 20 effective observations; since the
effective count under market weights is below the raw count, the switch appears
somewhat to the right of it.

**SUPP6** `SUPP6a_AllEmpiricalFits_Structure`, `SUPP6b_..._Envelope`,
`SUPP6c_..._Other`. One 15.5-in sheet of 147 panels becomes three pages, one
per material tier, at the printed width.

Before: ![](../archive/figures/CompareUQMethods_SUPP_AllEmpiricalFits.png)
After:
![](../outputs/figures/CompareUQMethods_SUPP6a_AllEmpiricalFits_Structure.png)
![](../outputs/figures/CompareUQMethods_SUPP6b_AllEmpiricalFits_Envelope.png)
![](../outputs/figures/CompareUQMethods_SUPP6c_AllEmpiricalFits_Other.png)

*Caption.* Every real EC3 category in the (a) structure, (b) envelope and (c)
other tiers, ordered by number of EPDs: a weighted histogram of the EPDs (gray),
the six fitted models, and the normalized mean of 1.0 (dotted).

**SUPP7** `SUPP7_ClaimScorecardMeanForm`. Shares Figure 3's cell, so it has
the same panel titles, legend and % labels, and its own tie boxes, computed on
the ratio of means.

![](../outputs/figures/CompareUQMethods_SUPP7_ClaimScorecardMeanForm.png)

*Caption.* As Figure 3, with every cell computed as the ratio of means, total
absolute error over total true value, and the tie test run on that statistic.

**SUPP8** `SUPP8_PLCATruth`. Panel labels, room between the title and the
panels, and a subtitle naming the orange tick.

Before: ![](../archive/figures/CompareUQMethods_FIG_PLCATruth.png)
After: ![](../outputs/figures/CompareUQMethods_SUPP8_PLCATruth.png)

*Caption.* The distribution over 2,500 synthetic pLCAs of each method's error
against the true market-weighted parents in (a) one material's contribution and
(b) the building total, where every material contributes 1.00 on average; the
orange tick is the mean error.

**SUPP9** `SUPP9_MixedPolicy`. The two single-statistic cutoff sweeps,
previously two separate files, are panels (a) and (b) of one figure, as you
asked.

Before: ![](../archive/figures/CompareUQMethods_FIG_MixedPolicy.png)
![](../archive/figures/CompareUQMethods_FIG_MixedPolicy_Median.png)
After: ![](../outputs/figures/CompareUQMethods_SUPP9_MixedPolicy.png)

*Caption.* Error pooled over fifteen claims against the cutoff (kernel density
estimate at or above, three-parameter lognormal below) for the size rule with
uniform weights (orange) and the same rule given the true market shares above
the cutoff (blue), as (a) the ratio of means and (b) the median of per-case
ratios. Gray lines are four fixed methods. Shading: the cutoffs
indistinguishable from the best on that statistic, 40 to 170 EPDs in (a) and 20
to 200 in (b). 2,500 synthetic pLCAs.

**SUPP10** `SUPP10_ChoiceDrivers`. The retired "equal weights", "market-share
weights" and "(eq)" are gone; the title says what is measured.

Before: ![](../archive/figures/CompareUQMethods_FIG_ChoiceDrivers.png)
After: ![](../outputs/figures/CompareUQMethods_SUPP10_ChoiceDrivers.png)

*Caption.* Share of the variation in which family is closer to the truth (the
log ratio of the kernel estimate's W1 to the three-parameter lognormal's)
explained on held-out synthetic datasets by dataset size alone (top bars and
dashed lines) and by size plus one other characteristic, under market (orange)
and uniform (gray) weights.

**SUPP11** `SUPP11_FlipCalibration`. **Its title said "a shift of one percent
of the mean ... changes the answer 5 percent of the time"; the 5% crossing in
its own table is 1.50% of the mean.** The number is now computed from the
table. Also %, panel labels, and no monospace.

Before: ![](../archive/figures/CompareUQMethods_FIG_FlipCalibration.png)
After: ![](../outputs/figures/CompareUQMethods_SUPP11_FlipCalibration.png)

*Caption.* (a) The chance that the leading material changes against the
relative W1 between two fitted models, for a calibration set of two weightings
of the same data (gray points, binned), the 15 pairs of the six UQ methods
(orange) and a logistic fit (gray line); triangles mark the 1, 5 and 10%
crossings. (b) to (d) Two densities as far apart as each crossing.

**SUPP12** `SUPP12_MaterialDominance`. %, panel labels, "1x 2x 5x" ticks as in
Figure 6, and the group size in a subtitle.

Before: ![](../archive/figures/CompareUQMethods_FIG_MaterialDominance.png)
After: ![](../outputs/figures/CompareUQMethods_SUPP12_MaterialDominance.png)

*Caption.* Against the ratio of the leading material's mean contribution to the
next, in four-material synthetic pLCAs: (a) the chance that switching UQ method
changes the leading material and (b) the change in a material's contribution.
Points are pLCAs; lines are a rolling mean (a) and median (b); the vertical
line is the 1% crossing at about 2.3 times.

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

Three numbers PRINTED ON SUPPLEMENT FIGURES were wrong against their own tables
and are corrected; no table value moved. SUPP5 drew and labeled the bandwidth
guard at 30, where the code uses 20; SUPP11's title said a shift of "one
percent" of the mean, where its 5% crossing is 1.50%; and SUPP5's lower row
called an in-sample W1 a W1 "against the target".

One quote in the narrative was mis-rounded and is corrected: the normal's
pooled median error is 26.447, printed as 26.5 in takeaways 1, 3 and 6 and in
decision 256; it is now 26.4 everywhere, as the graphical abstract prints it.

## 5. What is still open

- **The three qualifications in narrative section 9, item 1**, which the prose
  window must carry: F1, F2, and Figure 5's two ranges. Takeaway 8 says the
  kernel estimate "overtakes the lognormal on fit between 68 and 106
  EPDs". On the figure, 68 to 106 is the band of fit-level CUTOFFS
  indistinguishable from the best, and the families cross at 62 to 88
  (decision 142 treats these as different quantities).
- **Four labels in Figure 4 sit on shaded bands by design**, and
  `figstyle.check_overlaps` flags them. All four are legible at final size.
- **SUPP11's inset prints the flip crossings to four decimals** (0.0029, 0.0150,
  0.0316). Decision 175's rule for annotations rounds at the first digit where
  the two fits disagree, which would print "0.015 against 0.014" for the 5%
  level. Left as it is pending your view; it is the supplement.

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

**Figures:** `FIG1_` to `FIG9_` and `SUPP1_` to `SUPP12_` (SUPP6 is three
pages), PNG and PDF, the redrawn `FIG_GraphicalAbstract`, and every old
unnumbered pair in `archive/figures/`.

**Changed table:** `TABLE_GenerationExampleCurves.csv.gz`, the curves SUPP1
draws, now on a grid that ends where the data do; the drawn values beside it
are byte-identical. Shared labels: `src/dct_metriclabels.json` says "market
weights" and "uniform weights", and `comparison.base_label` strips both forms.

**Documents:** narrative section 4 (stems and captions) and section 9 (status),
`README.md` (figure list; `FIG7_WeightingBySize` was listed under notebook 1 and
is generated in notebook 2), `CONTEXT.md` (table inventory),
`archive/README.md`, `FIGURE_STYLE.md` (stem of its worked example), and
`CLAUDE.md` decision 257.

**Tests:** all pass (`python -m pytest tests/ -q`).

## 7. What the next stage picks up first

The prose, in its own window, per narrative section 9 item 2. It needs the
advisor-marked draft at an absolute path outside the repository, and it must
carry the three qualifications in section 5 above.
