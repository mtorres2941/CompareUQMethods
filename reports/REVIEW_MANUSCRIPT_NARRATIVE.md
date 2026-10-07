# Review of MANUSCRIPT_NARRATIVE.md

Reviewed 2026-10-07 against `outputs/tables/` on `corpus_2026-09-25` at
`weight_rho = 0.5`. Every command runs from the repository root after
`conda activate compareuq`. Most consequential first.

---

## 1. The Building 138 case study claims a run that no longer exists, and it says more than one building can show

The narrative says Building 138 was also run with every material at equal
intensity and gave "three different leaders" (takeaway 7, section 5 item 1,
section 7). That run was dropped by your decision of 2026-10-06, and nothing on
disk contains it. `TABLE_Building138.csv` has one setting, `real`. The
notebook cell says so: "The equal-intensity comparison is dropped (author,
2026-10-06)".

    python -c "import pandas as pd; print(pd.read_csv('outputs/tables/TABLE_Building138.csv').setting.unique())"

**The bolded "THE ONE REAL BUILDING ... SAYS THE 73 PERCENT IS CONSERVATIVE"
goes further than the measurement.** All seven policies naming the same leader
at 1.59x is roughly what the study's own calibration predicts for a
corpus-typical material. At four materials the chance that switching method
changes the leader crosses 10 percent at a lead of 1.53 and 5 percent at 1.73
(parametric fit), so at 1.59 agreement is expected about 90 to 95 percent of
the time anyway. One agreeing building cannot show that the 73 percent
overstates the risk. The narrower statement holds: a ready-mix class is tighter
(CV 0.24) than the corpus median material (CV 0.61), so a concrete-led building
plausibly sits on the safe side of the threshold. That is a hypothesis about
one building, not a finding.

    python -c "import pandas as pd; d=pd.read_csv('outputs/tables/TABLE_PLCARatioCrossings.csv'); print(d[d.nmats.astype(str)=='4'][['level','ratio','ratio_isotonic']])"

**And this building's lead material is outside the corpus.** All three
ready-mix classes in it (n = 14,366, 20,814 and 31,025) are above the corpus
maximum of 9,978. They are the three real categories the coverage figure lists
as uncovered on dataset size (finding 9).

## 2. Two headlines still carry ratio-of-means numbers under median-of-ratios wording

**Takeaway 4's heading, "worth four times what the rule is worth", is the
ratio-of-means ratio.** On the median of ratios, the headline statistic,
market shares are worth 4.40 points and the rule 0.14 points on the same pass.
That is about thirty times, not four. On the ratio of means it is 3.08 against
0.69, about 4.5 times.

    python -c "import pandas as pd; m=pd.read_csv('outputs/tables/TABLE_MixedPolicyScorecard.csv'); g=m.groupby('method'); print((g.median_error.mean()*100).loc[['Mixed','Feasible@80','KDE, Uniform']].round(2)); print((g.total_error.mean()*100).loc[['Mixed','Feasible@80','KDE, Uniform']].round(2))"

**Takeaway 1's heading says "wrong by about a quarter"** (ratio of means,
24.0). The thesis two sections above says "about a sixth (median 17)".

**Discussion item 1 says market-share data is "worth 3.08 of the 24 points"**
without saying that is the average building. On the headline statistic it is
4.40 of 17.4. A reader of section 1 and then section 3 sees two values for one
claim.

**Takeaway 8 says the whole cutoff sweep "moves the pooled error by 0.78
points"**, also unlabeled. That is the ratio of means. Figure 4's caption and
title say "about one point" (1.01 on the median). Same quantity, two values.

    python -c "import pandas as pd; d=pd.read_csv('outputs/tables/TABLE_MixedPolicyThresholdBothStats.csv'); x=d[d.family=='feasible'].groupby('statistic').pooled_error; print(((x.max()-x.min())*100).round(2))"

## 3. The cutoff band is stated three different ways, and Figure 4 shades the one the text says not to print

On the median of ratios the indistinguishable band in the 6,000-resample table
is **20 to 200**. Takeaway 3 says "20 to 300", which comes from the earlier
300-resample run, and cites the 6,000-resample table as if it agreed. Section
7.3 also says 20 to 300.

    python -c "import pandas as pd; d=pd.read_csv('outputs/tables/TABLE_MixedPolicyThresholdBothStats.csv'); x=d[(d.family=='feasible')&d.indistinguishable]; print(x.groupby('statistic').threshold.agg(['min','max']))"

The narrative says "Print 40 to 170" (takeaway 3, Figure 4, section 7.3).
`FIG_MixedPolicy_SpreadZoom` shades 20 to 200, the median band. Either the
figure or the instruction has to change. Open the PNG to see the shading.

Section 7.3 also says "Figure 4 is still drawn on the ratio of means and should
be redrawn with both curves". That was true of an older design. The chosen
SpreadZoom figure already draws both.

Takeaway 3's "so what" ("above about a hundred use a kernel estimate, below
about fifty a lognormal, in between it does not matter") contradicts the band
it sits under. If any cutoff from 40 to 170 performs the same, then the
datasets between 100 and 170 are also "does not matter", not "use a kernel
estimate".

## 4. Takeaway 8 prints a single-declaration threshold, which decision 225 forbids

"The kernel estimate overtakes the three-parameter lognormal on fit at **81
declarations**." Decision 225: "81 is not a number worth quoting at all ...
Nothing in the manuscript prints a single-declaration cutoff." Figure 5's
caption already gives the range (68 to 106). Takeaway 8 should do the same.

    grep -n "81" reports/MANUSCRIPT_NARRATIVE.md

## 5. Takeaway 7's per-comparison numbers are a mean labeled as a median, plus a figure from the superseded corpus

**"On one comparison the median absolute error runs 0.089 to 0.140"**: those
are the means. The medians are 0.052 to 0.115.

**"The six disagree about which design is better on 28 percent of
comparisons"** is decision 171's figure, measured on 2026-09-24, before the
corpus was regenerated. On the current table, at a claimed 5 percent saving,
the six methods fall on both sides of 0.5 on **41.6 percent** of design pairs.
The same definition gives 95.4 percent at a 0 percent saving, against
decision 171's 93, so it is the same quantity.

    python -c "import pandas as pd; d=pd.read_csv('outputs/tables/TABLE_PLCADesignSwap.csv.gz'); x=d[d.saving==0.05]; e=x.discernibility__error.abs().groupby(x.method); print(pd.DataFrame({'median':e.median(),'mean':e.mean()}).round(3)); p=x.pivot(index='pair',columns='method',values='discernibility'); print('disagree pct', round(100*((p>.5).any(axis=1)&(p<=.5).any(axis=1)).mean(),1))"

That makes "the decision a designer makes is robust" weaker than the heading
says, for designs that differ by a few percent.

## 6. The "17.23" benchmark in takeaway 3 is mislabeled, and the correct figure flatters the rule

Takeaway 3 says the rule's 17.38 compares with "17.23 for knowing in advance
which fixed method wins each claim". Picking the best of the three
uniform-weighted fixed methods claim by claim gives **17.44**, which is worse
than the rule. 17.23 is the best of four *including the rule itself*. So the
rule beats a reader who somehow knew the best fixed method for every claim.
That is a stronger statement than the narrative makes, and the current wording
inverts it.

    python -c "import pandas as pd; d=pd.read_csv('outputs/tables/TABLE_ClaimScorecardWithRule.csv'); p=d.pivot(index='claim',columns='method',values='median_error')*100; U=['KDE, Uniform','Lognormal, Uniform','Normal, Uniform']; R=[c for c in p if c.startswith('size')][0]; print('best fixed per claim',p[U].min(axis=1).mean().round(2),'| rule',p[R].mean().round(2),'| best of four incl rule',p[U+[R]].min(axis=1).mean().round(2))"

## 7. Building 138: the uncertainty-index sentence names the wrong method, and one mechanism claim is false

**Section 5 item 2 says ready-mix 5000 drives the uncertainty "under the
uniform-weighted lognormal, 35.2 against rebar's 35.7".** By those two numbers
rebar leads. In the table, the uniform-weighted *lognormal* has rebar ahead
(0.357 against 0.352). The one policy that puts ready-mix ahead is the
uniform-weighted *normal* (0.348 against 0.343).

    python -c "import pandas as pd; d=pd.read_csv('outputs/tables/TABLE_Building138.csv'); print(d.pivot(index='dataset',columns='method',values='ui').loc[['ReadyMix [5000-5999 psi]','RebarSteel']].round(3).T)"

**"Every dataset has a mean of 1.0, so a material's mean contribution is
exactly that intensity"** holds for the data and fails for the fitted models.
Under the normal fit, rebar's mean contribution is 59.2 against an intensity of
47.8, and brick's is 16.4 against 10.1. The truncation at zero raises the mean
of a wide normal. This is the mechanism behind item 3's 7 percent shift in the
total, and the text should say so rather than state the identity.

    python -c "import pandas as pd; d=pd.read_csv('outputs/tables/TABLE_Building138.csv'); print(d[d.method.str.startswith('Normal, U')][['dataset','intensity','eci_mean']].round(1))"

## 8. Two Discussion and Results numbers have no measurement behind them as stated

**Takeaway 11 (+0.83, 88.2 percent, 42.8 percent) has no reproduce command,
and these values appear nowhere in the repository except this file.** No
table, script or notebook produces them. By the review rule they are not
admissible until a command is attached. The section says they were
"recomputed on the shipped corpus", but no recomputation is recorded.

    grep -rnE "\+0\.83|88\.2 percent|42\.8 percent" CLAUDE.md CONTEXT.md reports notebooks audits src

**Discussion item 6 is contradicted by its own numbers.** It says a
uniform-weighted fit "flattens at 0.084 against a floor of 0.099 ... No
quantity of EPDs takes it below that floor". But 0.084 is already below 0.099.
At 1,000 or more declarations the mean uniform-weighted error is 0.0844, and
the mean distance between the two parents is 0.0995. Whatever the floor is,
this table does not show it binding. Restate it or drop it.

    python -c "import pandas as pd; d=pd.read_csv('outputs/tables/audits/TABLE_BandwidthNeff.csv'); print(d.groupby('band')[['uniform_best','parent_separation']].mean().round(4))"

## 9. Figure 2's caption claims full coverage that its own table and panels do not show

The caption says the corpus spans the real categories "on every characteristic
tested, leaving four uncovered dataset-metric pairs out of 1,470". The four are
real gaps on two characteristics. Coefficient of variation: Aggregates at 5.05
against a corpus maximum of 3.85. Dataset size: three ready-mix classes above
9,978. The figure plots two-dimensional pairs, and there it leaves 15 uncovered
points across six panels, six of them in one panel. The caption number is a
one-dimensional count describing a two-dimensional figure.

    python -c "import pandas as pd; d=pd.read_csv('outputs/tables/TABLE_MetricCoverage.csv'); print(d[d.empirical_covered<1][['metric','empirical_max','synthetic_max']]); c=pd.read_csv('outputs/tables/TABLE_CoverageFigureStats.csv'); print(c[['metric_x','metric_y','n_uncovered']]); print(c.n_uncovered.sum())"

## 10. Section 4's figure list is out of date with the rest of the file

- The table names `FIG_BuildingDominance_A, _B or _C` and
  `FIG_WeightingBySize_A, _B or _C`. Only the unsuffixed files exist, and the
  captions below already treat the designs as chosen.
- "Eight figures" omits `FIG_Building138`, which section 5 describes as "the
  figure" for the case study. Either it is a ninth main-text figure or it goes
  to the supplement, and the count should say which.
- Section 9.4's last bullet ("captions still quote ratio-of-means numbers") and
  section 9.1 are instructions to this review, now answered. Section 9.3 is
  history the file says it does not carry.

      ls outputs/figures | grep -E "BuildingDominance|WeightingBySize|Building138"

## 11. Small

- The Marsh et al. staircase at 1.02 is "near the tenth percentile" (takeaway
  7). It is at about the 3rd: the 10th percentile of the 292 buildings is 1.08.
  `python -c "import pandas as pd; r=pd.read_csv('data/raw/building_top2_benke2025.csv').top2_ratio; print(r.quantile(.1).round(3), round(100*(r<=1.02).mean(),1))"`
- "Wrong about twice as often as right" (takeaway 5) is true of the lognormal
  (32.8 percent closer) and not of the kernel estimate (38.5 percent, 1.6
  times). `python -c "import pandas as pd; print(pd.read_csv('outputs/tables/TABLE_MixedPolicyWeighting.csv')[['band','logn_market_closer_pct','kde_market_closer_pct']])"`
- The rule's margin over a kernel estimate everywhere appears as 0.16
  (takeaway 3, main pass: 17.38 against 17.54) and as 0.14 (takeaway 4, mixed
  pass: 17.42 against 17.56). Pick one pass for the comparison and say which.
  The commands in findings 2 and 6 print both.

---

## Checked and clean

- Every scorecard number in takeaways 1, 6, 9 and 10 and in Figure 3's captions
  matches `TABLE_ClaimScorecardWithRule.csv` or `TABLE_ClaimChoiceCost.csv`
  under the statistic it claims. That covers 7 of 15, the normal worst on 14 of
  15, 42.2 against 25.6, 41.8 to 47.3, the choice-cost percentiles, and the
  inversion at about 100 declarations on the median.
- **Figures 1, 2, 5 and 8 quote no ratio-of-means scorecard number.** Their
  numbers are dataset counts, fit-level W1 crossings or shape statistics.
  Figure 2's problem is a different one (finding 9).
- Takeaway 2's band gains, the 21.3 percent / 27 of 127 / 0.58 shape figures,
  Figure 7's table, the 73 percent and its 69.9 to 76.4 range, and the graphical
  abstract's 26 / 17 / 13 all reproduce.
- Every embedded image path exists in `outputs/figures/`.
