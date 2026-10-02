# Stage 4: the deposit

**Corpus `corpus_2026-09-25`. Empirical weight rule `WEIGHT_RHO = 0.5`.** Every
number below ran on those two and nothing else. Branch `stage-4-deposit`, from
`a28d322`. The working tree was already clean, so nothing needed committing
first.

---

## The first page

**The four code changes went in first and the run came once, which is the
order the Stage 3 review set (decision 239).** Notebooks 2, 3 and 4 then ran in
sequence: 14 minutes, 107 minutes, and notebook 4. Notebook 1 is unchanged by
this stage and was not re-run.

**The scorecard's count is 10 of 15, not the 11 decision 237 forecast, and the
reason is worth one sentence.** The rule is the best available choice on 10 of
the 15 claims; the claim that decides 10 against 11 is the design comparison,
where the rule reads 13.32 percent of the true level and the three-parameter
lognormal with uniform weights reads 13.21. They are a tie, and decision 245
anticipated this and told me to write whichever count the rebuilt table gives.

**Nothing moved that should not have, and that is measured rather than
asserted.** `TABLE_PLCAResults.csv` is byte identical over all 60,000 rows. The
four six-method truth tables are identical in content. The figure whose 720,000
random draws moved out of a figure cell comes back pixel identical. The control
decision 237 asked for reads **0.000e+00** on the numerator across all 90
(claim, method) cells.

**What needs a decision: nothing.** The figure numbering, full
`FIGURE_STYLE.md` compliance and the confidence intervals all wait on the
manuscript's figure selection, by decision 235, and this stage did not touch
them.

**Four judgment calls I made without asking**, each cheap to reverse: decision
241 is withdrawn as described in section 5; `WassVsResultDiff` is renamed to
carry a prefix; an exploratory notebook-3 cell is deleted; and the README's
description of the weights is rewritten because it was wrong in two ways.

---

## 1. The seventh scorecard column, on one pass

The feasible size rule now rides along on notebook 3's **main** truth run and
**main** design swap, so it and the six fixed methods are one Monte Carlo
experiment. It was previously scored in Stage 2j's own pass, and on the design
comparison the two passes resample their 2,500 design pairs independently.

**The control is exact where it can be.** Over the 90 (claim, method) cells of
the six: `error` and `error_portfolio` **0.000e+00**, `scale` 8.882e-16 and
`total_error` 1.110e-16, which is 3.00e-16 relative. `error` is computed WITHIN
one method so no other method can touch it; `scale` is a mean over seven
identical blocks instead of six, the same number by a different summation
order, and `total_error` is their ratio.

**The exact statement is pinned by a test rather than by the run**, because the
notebook's two paths differ in more than the seventh policy.
`test_adding_a_method_leaves_every_other_method_s_error_bit_identical` adds a
seventh method with genuinely different values and asserts every other method's
errors are unchanged, through both `recovery_table` and `per_unit_error`.

**AND THE CORRECTION MOVES THE COUNT THE OPPOSITE WAY FROM ITS OWN FORECAST.**
Decision 237 measured 11 of 15 and that number is real -- it is what you get
taking all seven columns from the MIXED pass, where the design comparison reads
13.38 for the rule against 13.40 for the lognormal. Taking all seven from the
MAIN pass, which is where the six methods the paper reports are measured, gives
13.32 against 13.21 and the rule loses that claim. **Both are internally
consistent; they disagree because the gap is about a tenth of a point against a
run-to-run spread of 0.70, which decision 237 measured itself.** What the fix
buys is not a different winner but a comparison that is one experiment.

**So what.** The figure's title is a count of claims, so until now one of those
claims was decided by which Monte Carlo run measured it rather than by which
method is better.

    python -c "
    import pandas as pd
    d=pd.read_csv('outputs/tables/TABLE_ClaimScorecardWithRule.csv')
    LAB='size rule\n(uniform)'
    p=d.pivot(index='claim',columns='method',values='total_error')*100
    CH=['KDE, Uniform','Lognormal, Uniform','Normal, Uniform',LAB]
    print(int((p[CH].idxmin(axis=1)==LAB).sum()),'of',len(p))
    print(p.loc['the probability B beats A',CH].round(2).to_dict())"

![The claim scorecard, seven policies on one pass](../outputs/figures/CompareUQMethods_FIG_ClaimScorecard.png)

*Fifteen claims, seven policies, all from one truth pass. Columns are grouped
by whether a reader can choose them: three uniform-weighted fits and the size
rule on the left, the three needing product-level market shares -- which nobody
publishes -- on the right. A solid box marks the closest of the four a reader
can choose, a dashed box the closest of all seven; the title counts the solid
boxes. Pooled over the fifteen claims the rule reads 23.96 percent of true
level against 24.66 for a kernel estimate with uniform weights and 24.73 for a
lognormal, so the margin behind that count is 0.7 points. The duplicated claim
of decision 238 is gone: a quantity reduction removes a deterministic fraction
of a material's own share, so its accuracy IS that share's accuracy and no
distributional assumption enters. Corpus `corpus_2026-09-25`,
`WEIGHT_RHO = 0.5`.*

## 2. Fifteen claims, not sixteen

"using 25 pct less: its mean saving" was exactly 0.25 times "a material: its
share of the total", so the scorecard counted one claim twice and every count
had a denominator of 16 where it should have been 15. `DUPLICATE_CLAIMS` in
`metricset.py` records what was removed, with the factor and the claim it
duplicates.

**Every pooled figure in the decision log moves with the denominator and no
measurement does.** Pooled over fifteen claims rather than sixteen: the rule
23.96 against 23.25, the kernel estimate with uniform weights 24.66 against
23.92, the lognormal 24.73 against 23.96, the kernel estimate with market
shares 23.58 against 22.89. The ORDERING is unchanged and so is the count, 10
either way, because the dropped claim was a loss for the rule in both.

**So what.** A reader counting boxes on that figure was being shown one claim
twice, and both copies were losses for the rule a practitioner can follow.

    python -c "
    import pandas as pd, sys; sys.path.insert(0,'src'); import metricset
    print(len(metricset.SCORECARD_CLAIMS), 'claims;', metricset.DUPLICATE_CLAIMS)"

## 3. The group-composition split carries both rules

Stage 2j built this split for the known-share rule and left the feasible rule's
column for "the next full run". Two full runs went by because widening it is a
code change. Pooled relative error over every claim belonging to a pLCA group,
by how many of the group's four materials the rule moves:

    materials                        Lognormal  KDE       KDE       Feasible
    above the cutoff   groups        uniform    uniform   market    @80       Mixed
    0 of 4                117         0.3373    0.3494    0.3806    0.3373   0.3373
    1 of 4                572         0.2992    0.3107    0.3266    0.2962   0.2828
    2 of 4                948         0.2605    0.2597    0.2547    0.2529   0.2245
    3 of 4                690         0.2180    0.2071    0.1714    0.2041   0.1557
    4 of 4                173         0.1674    0.1541    0.0878    0.1541   0.0878

**Each rule is now checked against the two fixed methods it collapses to at its
own ends, and all four controls read 0.00e+00**: `Mixed` against
`Lognormal, Uniform` at 0 of 4 and against `KDE, Variable` at 4 of 4;
`Feasible@80` against `Lognormal, Uniform` at 0 and against `KDE, Uniform` at
4. Everything either rule buys is in the middle, and the feasible rule's gain
peaks where two of four materials move, which is also the commonest
composition.

**So what.** The rule's gain is attributable rather than merely present: a
probabilistic LCA claim belongs to the group of four, so improving one material
cannot improve the claim by more than its share of it, and the table shows
exactly that shape with exact zeros at both ends.

    python -c "
    import pandas as pd
    print(pd.read_csv('outputs/tables/TABLE_MixedPolicyPooled.csv').to_string(index=False))"

## 4. All 37 of 37 figure cells now redraw on their own

Notebook 3 had nine of thirteen figure cells that could not be rendered without
the whole run, and it was believed clear because every use of the renderer had
passed `--only` and drawn a cell that happened to be self-sufficient.

**The serious find is that a figure cell was consuming the notebook's random
stream.** `FIG_PLCAVisualizeUQFits` drew its whole-building totals with
`rvs(..., random_state=rng)` inside the figure cell -- 72 blocks of 10,000
draws -- so redrawing that figure moved every number after it. The draws are in
a compute cell now, in the same place in the stream. **It comes back pixel
identical: 0 differing pixels of 4,596,519.**

The other three changes are mechanical: the display constants and the two
comparison helpers move into the setup cell, the helpers take their frames as
arguments, and five frames that lived only in kernel memory are persisted.

**One defect the refactor introduced, and the split is what made it cheap to
find and fix.** A pivot carries its key names onto the frame and seaborn turns
those into axis labels, so two case-study figures came back with `pewt1` and
`pewt2` printed over the dataset titles. One `rename_axis` each, redrawn
through the renderer in seconds rather than through another 107-minute run, and
**both are now 0 differing pixels against the pre-refactor images**.

**So what.** A figure round in notebook 3 was the full run, 107 minutes. The
heaviest of its figures, a scatter of about 1.8 million points, now redraws in
**27 seconds**, measured with another notebook competing for the processor.
The two case-study figures above redrew in under five.

    python audits/figure_manifest.py
    python -m pytest tests/test_figure_manifest.py -q
    python audits/render_figures.py 03_CompareUQ_PerformPLCA --only "heatmaps of W1" --out /tmp/figs

## 5. What the Stage 3 review got wrong, and what was wrong instead

**Decision 241 asks this stage to add (a) (b) (c) (d) panel labels to the pLCA
scatter figure. They have been there since 2026-05-12.**

    git log -S "alphabet[ires]" --oneline -- notebooks/03_CompareUQ_PerformPLCA.ipynb

The committed PNG shows them. What IS wrong with that figure, and is visible in
the same image, is that the top row's x axis labels print on top of the bottom
row's two-line titles. The row spacing is fixed instead.

**So what.** Nothing was owed here and a real defect sat next to the one that
was reported. The review formed the finding from the prompt's instruction
rather than from the figure; an image is reproduced by looking at it.

## 6. Numbers that moved

| what | before | after | why |
|---|---|---|---|
| the scorecard's claim count | 16 | **15** | the duplicated claim, decision 238 |
| the rule is the best available choice on | 10 of 16 | **10 of 15** | the dropped claim was a loss for the rule, so the count is unchanged |
| pooled error, the rule | 23.25 | **23.96** | the denominator, 15 claims not 16 |
| pooled error, KDE with uniform weights | 23.92 | **24.66** | same |
| pooled error, lognormal with uniform weights | 23.96 | **24.73** | same |
| pooled error, KDE with market shares | 22.89 | **23.58** | same |
| the design comparison, the rule | 13.38 | **13.32** | measured on the main pass instead of the mixed pass |
| `TABLE_MixedPolicyThreshold` pooled error | -- | moves by at most 7.7e-3 | the 15-claim denominator |
| `TABLE_MixedPolicyRanking` `n_claims` | 16 | **15** | same |
| `TABLE_WeightingOnCommonTarget` columns | `variable`, `variable_wins` | `market`, `market_wins` | the retired vocabulary, decision 199 |
| `TABLE_WeightingDecomposition` column | `rank_vs_variable_target` | `rank_vs_market_target` | same |
| `TABLE_ReductionWeightConcentration` column | `variable_closer_pct` | `market_closer_pct` | same |

**Nothing else moved, and every place it could have is checked.**

**Notebook 2.** Of 168 tables, **159 byte identical and 7 more identical in
content** -- a gzip header timestamp or a `written_utc` field. The two that
changed are the column renames above, same shape and same values. Of 55
figures, **two PNGs changed and both changed only text**:
`SUPP_AllEmpiricalFits` differs in 4,932 of 21,058,587 pixels, all inside one
legend line now reading "gray other", and `FIG_TargetBySize`'s third panel
asked "Does variable weighting help?" and now asks "Do market weights help?".
The 36 PDFs that differ differ in their embedded creation timestamp.

**Notebook 3.** `TABLE_PLCAResults.csv` **byte identical** over 60,000 rows;
the four six-method truth tables identical in content; 145 tables byte
identical and 13 more identical in content. **Eleven tables are new** -- four
carrying the rule's rows, seven backing figure cells that used to read kernel
state -- and the nine that changed are the scorecards losing one claim's worth
of rows, the pooled table gaining two columns, and the two threshold tables
moving with the denominator.

## 6b. Where this report is most likely wrong

**For the window whose job is to attack it.** Three places I would start.

**The count is 10 of 15 and decision 237 forecast 11.** Section 1 says both
numbers are real and names the claim that separates them. Check that the
explanation survives the two tables, and in particular that 13.32 against 13.21
is read off the MAIN pass and 13.38 against 13.40 off the MIXED one.

**"Nothing else moved" rests on a comparison script I wrote**, not on a tested
one. It reports byte equality, then frame equality, then the largest numeric
move per column. It does not compare row ORDER within a frame of equal shape
and equal values, so a reordering would read as identical.

**The pixel comparisons are mine too.** "0 differing pixels" is a real
statement about two PNG files; it is not a statement about the PDF siblings,
which differ in every case because they carry a creation timestamp.

## 7. The deposit

**Which corpus the paper describes.** A reader arriving from the citation saw
49 `corpus_*` directories and nothing saying which one the study used.
`data/processed/README.md` says it: **`corpus_2026-09-25`**, named by
`CORPUS.json`. The rest are superseded and kept only because a corpus here is
immutable, so a change to generation writes a new one beside the old and the
two can be diffed -- which is how a change is proved to have moved only what it
was meant to. It also says which files are tracked and how to rebuild the
200 MB ones. `corpus_2026-09-15b`'s two derived replay caches are untracked:
6.6 MB of cache for a superseded corpus whose data the deposit does not carry,
and `.gitignore` explicitly left that removal to this stage.

**THE README'S DESCRIPTION OF THE WEIGHTS WAS WRONG IN TWO WAYS.** It said both
arms draw market shares from a flat Dirichlet. Since decision 190 the empirical
arm uses `weighting.coherent_weights` at a coherence of 0.5, and on the
synthetic arm the share attached to a product group is that group's TRUE share
in the parent (decision 212) -- so uniform against market there is ignoring a
KNOWN share against using it, not guessing against knowing.

**So what.** That is the distinction the whole weighting comparison rests on,
and it is the objection you raised in Stage 2j. A reader checking the
repository against the paper would have drawn the wrong conclusion from the
README alone.

**The figure manifest is in the README itself**, because the audit table it
comes from is gitignored and the deposit would not carry it. 55 images grouped
by notebook, each with what it shows, and no figure numbers (decision 235).
`WassVsResultDiff` was the one image with neither prefix and is now
`FIG_WassVsResultDiff`; the old pair is in `archive/` with the reason.

**The spelling and vocabulary sweeps are finished**, which closes the item
Stage 2g handed to the deposit tidy-up. Fourteen files, all comments and prose.
`characterisation` survives in one place, the published title of Marsh, Lewis,
Hattam and Allen (in press), and the four files with non-ASCII characters are
the justified ones. The retired vocabulary is out of the column names and out
of one figure the Stage 3 sweep missed. **The stored `method` values keep
"Variable"** by decision 199, so a pivoted table still shows `KDE, Variable` as
a column header: that is the join key, not a label.

    python -m pytest tests/test_figure_manifest.py -q
    grep -rniE 'colour|centre|labelled|neighbour|grey' --include='*.py' --include='*.md' src/ audits/ tests/ *.md

## 8. The superseded reports are deleted

CLAUDE.md keeps only the CURRENT stage's report. Six `HANDOFF_stage-*.md`
files, `STAGE_REPORT_2j.md` and `STAGE_REPORT_3.md` are deleted; git history
retains them, which is what makes it safe.

**Nothing outstanding lived only there, and that was checked per item rather
than per file.** The Stage 2h handoff carries the cumulative open list and
every item on it is either resolved in the decision log or is this stage's own
work; the two it assigned to "the deposit tidy-up" are the British spellings
and the retired vocabulary, both done. Stage 2j's open items were settled by
decision 207 and Stage 3's by decision 235 and the Stage 4 prompt. The one
manuscript item that lives in neither -- restating the coverage claim from the
rebuilt table -- is entry 34 of `reports/MANUSCRIPT_discrepancies.md`, which
survives.

**`reports/STAGE_PROMPTS.md` is now a record in full** and no section may be
edited. The only part still maintained is the configuration block at the top,
which this stage updated with its two production changes.

## 9. What is still open

**Three items, and all three wait on the same thing: which figures the
manuscript carries.** None is repository work and none is waiting on a
measurement.

| item | why it waits |
|---|---|
| **The figure numbering.** 55 images, no numbers. `TABLE_FigureManifest.csv` lists every one with its generator and the README carries the manifest; `figstyle.savefig` takes a stem, so applying a numbering is one word per figure cell and the duplicate-name guard makes a collision impossible | decision 235: numbering needs the selection, the selection needs the manuscript's structure |
| **Full `FIGURE_STYLE.md` compliance for the 29 figure cells that never call `figstyle.apply()`.** They carry the palette, type sizes and spine rules of whatever they were written with. The ASCII-minus half is done | the same selection. A redesign of 29 figures when six reach the paper is the wrong order |
| **Confidence intervals on figure aggregates.** 13 candidates, several of them scatters that need none | which aggregates need one is a question about what the paper claims |

**And one paragraph the manuscript owes rather than an open task**: the
coverage claim restated from the rebuilt table, entry 34 of the discrepancy
file.

## 10. Inputs and outputs

**Read.** `corpus_2026-09-25`; the frozen EC3 extract; every table under
`outputs/tables/`.

**Written.** Nine new tables -- four carrying the size rule's rows from the
main truth and swap passes, and five backing notebook 3's figure cells; the
figure manifest and the renderer-safety list; `data/processed/README.md`; the
figure manifest section of the README; and the manuscript-limitation checklist
in CLAUDE.md.

**Code.** `metricset.SCORECARD_CLAIMS` loses the duplicated claim and gains
`DUPLICATE_CLAIMS`; notebook 3's setup cell gains the display constants, two
loaders and the two comparison helpers; two tests are new.

**Reproduce.** Notebooks 2, 3 and 4 in order. Notebook 1 is unchanged by this
stage and was not re-run.

## 11. What the manuscript session picks up first

**This is the last stage, so there is no next one.** Three things belong to the
manuscript:

1. **The figure selection, then the numbering, then the style pass and the
   confidence intervals** -- one pass over six or seven figures rather than
   three passes over 55. Section 9 and decision 235.
2. **The limitation checklist in CLAUDE.md**, which names each Discussion
   paragraph this rework retires, what replaces it, and the decisions that did
   it: six inversions and three new limitations the rework adds.
3. **`reports/MANUSCRIPT_discrepancies.md`**, the list of places the manuscript
   and the code disagree.
