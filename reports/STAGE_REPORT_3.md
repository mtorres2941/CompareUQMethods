# Stage 3: figures

**Corpus `corpus_2026-09-25`. Empirical weight rule `WEIGHT_RHO = 0.5`.** Every
number below ran on those two and nothing else; where a figure or an audit was
re-run, it was re-run on them. Branch `stage-3-figures`, from `97acefd`.

---

## What changed

**A figure round is now seven seconds instead of thirty-five minutes.**
`audits/render_figures.py` refused notebooks 1 and 2 outright, and two tests
skipped saying so. There were two blockers, not one: neither notebook defined
`OUT`, and their figure cells read fitted models and frames out of kernel
memory. Both are cleared. Every figure cell in all four notebooks now reads a
table from disk.

**It moved nothing, and that is checked rather than claimed.** Notebook 1 re-ran
end to end: all 147 x 24 empirical characteristic values are bit identical to
the committed table, and four of its five figure PNGs are byte identical. The
fifth changed one axis label. Notebook 2's synthetic score table is bit
identical over 10,000 rows and 39 columns.

**Figures 2 and 3 are one figure**, `FIG_W1DistanceAndRank`: the W1 strip on the
left, the rank-frequency heatmap on the right, a row per arm. **Figure 4 is
rebuilt** to the four characteristics Stage 2f's reduction kept, uniform and
market weights side by side in a row, log y axis, rows ordered by importance
read from the reduction table. **The three-curve alternative you asked to
compare it against is beside it**, `FIG_WeightingGap_*`.

**The scorecard has its seventh column** and the rule is closest of the four
options a reader has on 10 of 16 claims.

**Five figures had no generator anywhere** and are in `archive/` with reasons.
Three tests now fail on a duplicate filename, an orphan, or a retired weighting
word in a filename.

## What needs a decision

**The figure numbering, which is below and which I have NOT applied.** You asked
to confirm before renaming, and 49 figures is more than any paper carries, so
the proposal is as much about which figures are main text as about numbers.

**Nothing else.**

---

## 1. The two blockers, and the control that says the split was safe

`tests/test_render_figures.py` skipped notebooks 1 and 2 with the reason "has no
setup cell". Marking their figure cells would have unlocked nothing: the tool
raises on a missing `OUT` before it ever looks at a marker.

Clearing it needed the separation of compute from plotting that this project
requires anyway. Where a figure needed something never persisted -- the example
parents, the six fitted densities, the KS and W2 scores, every empirical fit --
the cell is split into a compute half that writes a table and a figure half that
draws it.

**The split preserves the random stream exactly, which is why no number moved.**
Notebook 1's example figures draw from the main Generator and
`corpus.make_combos` later takes the pLCA groupings from that same stream, so
moving a draw would have moved every pLCA number. The compute halves consume
what the single cells consumed, in the same order.

**So what.** Moving a label on a figure used to cost half an hour, so labels did
not get moved. They do now.

    python -m pytest tests/test_render_figures.py tests/test_notebooks.py -q
    python audits/render_figures.py 01_CompareUQ_CreateData --out /tmp/figs
    git diff --stat HEAD~2 -- outputs/tables/TABLE_EmpiricalECCMetrics.xlsx

## 2. The scorecard's seventh column, and the two counts it carries

The feasible rule -- uniform weights throughout, kernel estimate above the
cutoff and three-parameter lognormal below -- is now a column on
`FIG_ClaimScorecard`, never the known-share rule (decision 211).

**It is closest of ALL SEVEN on 2 of 16 claims and closest of the FOUR A READER
CAN CHOOSE on 10 of 16.** Both are true and they are different questions. The
three market-weighted columns need product-level shares nobody publishes, so
they carry an asterisk and one footnote, and the figure draws two boxes: black
for the best of seven, dashed for the best of the four.

**10 is not decision 217's 9 and the paper must not use one for the other.**
Mine is a plain argmin; 217's is the rule's gain over the best uniform-weighted
method with a paired interval excluding zero. They differ on exactly one claim,
a material's standard deviation, where the rule is closest by 0.52 percent and
the interval runs -0.37 to +1.46.

Pooled over the sixteen claims: the rule 23.25 percent of true level, the
kernel estimate with uniform weights 23.92, the three-parameter lognormal with
uniform weights 23.96, and the kernel estimate with market weights -- which
nobody can use -- 22.89.

**So what.** A reader with a pile of EPDs and nothing else can count them and
pick a curve, and on most of what a probabilistic LCA reports that beats any
single method they could have fixed on.

    python -c "import pandas as pd; d=pd.read_csv('outputs/tables/TABLE_ClaimScorecardWithRule.csv'); print(d[d['rank']==1].method.value_counts())"
    python audits/render_figures.py 03_CompareUQ_PerformPLCA --only "every claim"

![The claim scorecard, seven policies](../outputs/figures/CompareUQMethods_FIG_ClaimScorecard.png)

*Each cell is the mean absolute error against the true parent per unit, as a
percentage of the true level of the same quantity. Under the best of the seven
a probabilistic LCA is right to 12.6 percent on the design comparison and wrong
by 32.4 percent on which material leads. Corpus `corpus_2026-09-25`,
`WEIGHT_RHO = 0.5`.*

## 3. The hump-spacing measurement, which nobody had made

Both levers built for the joint modality-and-dispersion gap --
`separation_dispersion_frac` and `shoulder_frac` -- were still at 0.0 and had
never been measured under the settled weight rule. Each is now measured ALONE
against the shipped configuration, 1,000-dataset drafts, judged against the
**weight-draw noise of 0.006 to 0.015**:

    candidate               multimodal  both  multi|dispersed  objective  weighting
    shipped                     0.200  0.013            0.147     0.2281     0.1722
    + separation                0.233  0.022            0.191     0.2730     0.1520
    + shoulder                  0.167  0.019            0.167     0.2214     0.1420
    the real arm                0.246  0.054            0.212         --         --

**Separation closes most of the conditional gap, 0.147 to 0.191 against a real
0.212, and costs 0.045 on the objective, which is three to seven times the
noise.** Shoulder is free on the objective, -0.0067, buys about half the joint
gap, and gives up the multimodal marginal, 0.200 to 0.167.

**And the old trade is confirmed gone.** Both candidates IMPROVE the weighting
margin, 0.1722 to 0.1520 and 0.1420, where every pre-port measurement had
widening cost it. Decision 193 is confirmed on a lever it was not measured on.

**Recommendation: do not regenerate.** Neither lever closes the gap, both have a
price, and decision 203 already settled the limitation. What is new is that it
now rests on a direct measurement of the two levers rather than on inference
from a third.

**A defect found doing it.** The audit reused any existing draft directory, and
the `current` draft had been generated under the SUPERSEDED configuration --
`cv_log10_mean` 0.129 against 0.329, `min_q1_over_iqr` 0.5 against 0.2 -- so
every candidate was being compared against the wrong baseline while the table
said `current`. It now refuses a draft whose recorded configuration differs
from the one asked for.

    python audits/corpus_joint_structure.py 1000 --only current,shipped_plus_separation,shipped_plus_shoulder

## 4. The stale audits, re-run

All six re-ran on the current corpus -- the generator sweep was already re-run
in Stage 2h (decision 200), so five ran here. **Every ordering held and every
level moved**, which is what an absolute distance does when the corpus gets more
dispersed.

**The upper truncation.** The worst runaway fit is 4.95 times the data's own
spread uncapped, not the 5.4 decision 182 records, and a cap at twice the
largest observation takes it to 1.85. Against the truth the cap still costs
nothing: `Lognormal, Variable` 0.19924 uncapped against 0.19913 capped, and the
kernel estimate and the normal do not move at all. **Decision 199's one-line
remedy stands and its numbers need updating from 5.4 and 1.8 to 4.95 and 1.85.**

**The judgment arm.** The six data-driven methods span 0.0756 to 0.1140 on the
design comparison; a correctly centred pedigree model at matched spread reads
0.0886, and 0.0886 to 0.1136 across a six-fold spread range. A common offset
cancels to the last digit, which reproduces decision 184's control exactly. The
realistic case -- the centre taken from one declaration -- reads 0.1612 to
0.2722. At matched spread and correct centre, uniform reads 0.2349 and
triangular 0.2228 against the pedigree lognormal's 0.0886, so the shape finding
is now 2.5-fold rather than 2.1-fold.

**The certification credit.** At a 10 percent credit demonstrated with 75
percent confidence the truth earns it on **12.7 percent** of 3,000 cases, at
least two of the six methods disagree on **19.4 percent**, the best method calls
it wrong on **7.5** and the worst on **13.0**. **The fragility is still the
threshold and not the methods**: the six disagree on 59.1 percent of designs
whose true confidence sits within 0.05 of the line, 54.0 percent between 0.05
and 0.10, 23.4 percent out to 0.25 and 5.1 percent beyond it. A normal is the
worst method on **9 of the 15** tier-and-confidence combinations, where decision
187 records 8.

**The bandwidth sweep.** The two arms still disagree about the density criterion
and agree about the study's own: on the synthetic arm Scott beats the other two
on held-out likelihood 64.0 percent of the time under market weights, while on
W1 Silverman reads 0.0716 against Scott's 0.1244. Decision 188 holds.

**The bandwidth through the pLCA, and this one NARROWS decision 195.** Averaged
over the five outputs, Scott is still the worst rule under both weightings --
29.23 against Silverman's 28.24 and the shipped guarded rule's 28.51 under
uniform weights, 30.06 against 28.89 and 28.93 under market weights -- so the
shipped rule stands and nothing changes. **What does not survive is "Scott is
worst on every one of the five outputs":** under uniform weights Scott is now
best on two of them, a material's standard deviation at 31.091 against the
guarded rule's 31.035 and the uncertainty index at 49.412 against 49.428. And
**the guard no longer buys anything under market weights**, where decision 195
records it buying 0.15 of a point; it now costs 0.04 there and 0.28 under
uniform weights. Both are small and neither changes the choice. The control
fires: the four parametric methods move 0.0000 points across the three rules.

    bash -c 'for a in upper_truncation bandwidth_rules judgment_arm credit_design bandwidth_downstream; do python audits/$a.py; done'

## 5. The proposed figure numbering, which needs your yes

**Six main-text figures**, following the five questions a probabilistic LCA
answers. Everything else becomes supplement.

| new | current stem | what it says |
|---|---|---|
| FIG1 | `FIG_PDFandCDFofUQMethods` | what the six methods are |
| FIG2 | `FIG_W1DistanceAndRank` | how far each sits from the data, and how often it wins (the merge) |
| FIG3 | `FIG_WhenToUseWhich` | which method is closest, against category size |
| FIG4 | `FIG_W1VsSurvivors_Synthetic` | where each method sits against the characteristics that carry signal |
| FIG5 | `FIG_ClaimScorecard` | every claim, all seven policies, against the truth |
| FIG6 | `FIG_MixedPolicy` | how much the cutoff matters, and what market shares are worth |

`FIG_WeightingDrivers` is the strongest practitioner-facing figure in the
project and is the obvious seventh if you want one. `FIG_WeightingGap_*` is the
alternative to FIG4 -- three curves instead of six -- and I have drawn both so
you can choose; I would keep FIG4 and put the gap version in the supplement,
because the levels are what a reader needs first.

The remaining 43 images become `SUPP<N>_*`. The full list, with the generator of
every one, is `outputs/tables/audits/TABLE_FigureManifest.csv`.

**Nothing is renamed until you say so.** `figstyle.savefig` already takes a stem
rather than a path, so applying the numbering is a one-line change per figure
and the duplicate-name check makes a collision impossible.

    python audits/figure_manifest.py

## 6. Numbers that moved

| what | before | after | why |
|---|---|---|---|
| `flip.prose_crossing` at 5 pct | 0.02 against 0.01 | 0.015 against 0.014 | decision 175's rule rendered two values 7.5 percent apart as a factor of two, because 0.0150 sits on a rounding boundary. The printed spread may now be at most twice the real one. Both of decision 175's own published examples are unchanged |
| `flip.prose_crossing` at 10 pct | 0.032 against 0.031 | 0.0316 against 0.0314 | same guard |
| upper-truncation worst fitted spread | 5.4 | 4.95 | re-run on the current corpus |
| upper truncation at 2x | 1.8 | 1.85 | same |
| judgment arm, pedigree at matched spread | 0.097 | 0.0886 | same |
| judgment arm, uniform / triangular | 0.206 / 0.198 | 0.2349 / 0.2228 | same |
| the credit: truth earns it | 17.4 pct | 12.7 pct | same |
| the credit: methods disagree | 18.2 pct | 19.4 pct | same |
| the credit: normal is worst on | 8 of 15 cells | 9 of 15 | same |
| bandwidth through the pLCA, mean over five outputs | 25.47 / 24.82 guarded | 28.51 / 28.93 | same |
| `TABLE_MethodCurves` | 96.4 MB csv.gz | 44.4 MB parquet | container only; every float still float64 and every row verified element by element before the CSV was removed |
| `TABLE_MethodWinShare` | 22.8 MB | 6.7 MB | same |
| `audits/judgment_arm.py` column `centre` | `centre` | `center` | the US spelling sweep; the table is regenerated under the new name |

**Nothing else moved.** The empirical and synthetic characteristic tables, the
six W1 score columns and every pLCA table are bit identical.

## 7. What is still open

| item | |
|---|---|
| **The figure numbering needs your confirmation before anything is renamed.** Section 5 |
| **30 of 36 marked figure cells do not call `figstyle.apply()`**, so they follow the palette, the type sizes and the spine rules of whatever they were written with. The ASCII-minus requirement is now met at notebook level, which was the correctness half; the rest is a redesign of 30 figures and is a Stage 4 or manuscript-session job. `outputs/tables/audits/TABLE_FigureStyleCompliance.csv` names them |
| **`FIG_MetricCoverage` still needs its claim restated in the text** from the rebuilt table: the figure is current, the sentence in the manuscript is not |

## 8. Inputs and outputs

**Read:** `corpus_2026-09-25`; the frozen EC3 extract at
`data/raw/ec3_raw_ecc_2026-08-14.csv.gz`; every table under `outputs/tables/`.

**Written:** eleven new tables backing figure cells that used to read kernel
state (`TABLE_GenerationExample*`, `TABLE_DatasetExample*`, `TABLE_W1Demo*`,
`TABLE_MethodDemo*`, `TABLE_SyntheticKSandW2`, `TABLE_MethodRankFrequency`,
`TABLE_EmpiricalValues`, `TABLE_EmpiricalFitCurves`); `TABLE_ClaimScorecardWithRule`;
`TABLE_FigureManifest` and `TABLE_FigureStyleCompliance`; a `.pdf` beside every
PNG a re-run notebook wrote.

**Code:** `figstyle.savefig` and the duplicate-name guard; `metricset.rescore`;
the prose-rounding guard in `flip`; `audits/figure_manifest.py`; the stale-draft
guard in `audits/corpus_joint_structure.py`; `tests/test_figure_manifest.py` and
three new tests in `test_notebooks.py` and `test_flip.py`.

**Reproduce the whole stage:** notebooks 1, 2 and 3 in order, about 30, 35 and
195 minutes. Notebook 3 has not been re-run end to end in this stage; the
scorecard figure and the seven-policy table were produced by the figure renderer
against the tables already on disk, which is what that tool is for.

## 9. What Stage 4 picks up first

1. **The figure numbering, once confirmed.** It is a stem change per cell plus a
   manifest regeneration, and the duplicate check will catch any collision.
2. **Re-run notebook 3 end to end.** Stage 2j left two measurements for the next
   full run -- the feasible rule's group-composition split, and `MixedBackwards`
   as a control that should lose to both recommended rules -- and the pLCA
   scatter figure's highlighted points were dropped this stage, which shifts the
   illustrative sampling in `FIG_PLCAVisualizeUQFits` and nothing else.
3. **The README and the deposit**, which is Stage 4's own work, now with a
   figure manifest to put in it.
