# archive/

Figures that no code in this repository produces any more. **Moved here rather
than deleted**, on the author's standing instruction, so that nothing is lost
and the move is one `git mv` from being undone.

Each one was found by `python audits/figure_manifest.py`, which walks every
notebook cell and every module and joins what they write against what is on
disk. A file in this directory had no generator at all.

| file | why it is here |
|---|---|
| `CompareUQMethods_FIG_Wass1DistStripAndRank.png` | superseded by `CompareUQMethods_FIG_W1DistanceAndRank.png`, which merges the W1 strip and the rank-frequency heatmap into one figure on the author's Stage 3 instruction |
| `CompareUQMethods_FIG_WassRankFreq_Heatmap.png` | the other half of the same merge |
| `CompareUQMethods_FIG_WhatMattersForTheChoice.png` | notebook 4, dated 2026-09-21. Superseded by `FIG_ChoiceDrivers` and `FIG_WhenToUseWhich` when the Stage 2f review moved the reduction out of sample and onto the synthetic arm |
| `CompareUQMethods_SUPP_ChoiceDrivers_Synthetic.png` | same date, same cause |
| `CompareUQMethods_SUPP_ChoicePowerByArm.png` | same date, same cause |
| `CompareUQMethods_FIG_BuildingDominance_histogram.png` and `.pdf` | the single histogram design, rejected by the author on 2026-10-06 ("histograms are the poster child of binning"). Renamed with a suffix so it cannot be mistaken for the current `FIG_BuildingDominance`, the sorted-dot design the author chose |
| `CompareUQMethods_FIG_BuildingDominance_A.png`, `_B.png` and `.pdf` | the two designs NOT chosen for the same figure, 2026-10-06: A the empirical CDF, B a box plus strip. The author chose C |
| `CompareUQMethods_FIG_WeightingBySize_histogram.png` and `.pdf` | same date and same reason: its lower panel was a histogram, and it drew a line at 80 where the weighting split is published as the band 80 to 100 (decision 225). Renamed with a suffix so it cannot be mistaken for the current `FIG_WeightingBySize`, which is the box-plus-strip design the author chose |
| `CompareUQMethods_FIG_WeightingBySize_A.png`, `_C.png` and `.pdf` | the two designs NOT chosen for the same figure, 2026-10-06: A on a continuous axis of declarations, C the rolling win-share curve. The author chose B |
| `CompareUQMethods_FIG_GraphicalAbstract_B.png`, `_C.png` and `.pdf` | the two graphical-abstract designs NOT chosen, 2026-10-06: the bars alone and the rule as a number line, both "too simple" in the author's words |
| `CompareUQMethods_FIG_Building138_A.png`, `_B.png`, `_C.png` and `.pdf` | the first three case-study plots, 2026-10-06. A compared the real building with an equal-intensity version the author judged not to make sense; C drew the building total, which he did not find informative; B's uncertainty-index dots survive inside the ridgeline that replaced all three |
| `CompareUQMethods_FIG_Building138_ByUncertainty.png` and `.pdf` | the case-study ridgeline with rows ordered by uncertainty index, 2026-10-06. The author chose rows by contribution, now `FIG_Building138` |
| `CompareUQMethods_WassVsResultDiff.png` | RENAMED rather than superseded, in Stage 4. It was the one image in the repository carrying neither the `FIG_` nor the `SUPP_` prefix, and its own cell declares itself a figure, so it is written as `CompareUQMethods_FIG_WassVsResultDiff` from the next run of notebook 3. The content is unchanged |
| `CompareUQMethods_FIG_PDFandCDFofUQMethods`, `_MetricCoverage`, `_ClaimScorecard`, `_MixedPolicy_SpreadZoom`, `_WhenToUseWhich`, `_BuildingDominance`, `_WeightingBySize`, `_ShapePlane`, `_Building138`, each `.png` and `.pdf` | RENAMED with their manuscript number in the figure pass of 2026-10-07 (`reports/REPORT_FIGURE_PASS.md`): the same nine figures are now written as `FIG1_` to `FIG9_`. Kept here as the before-image of that pass; `FIG6_BuildingDominance.png` is byte-identical to its old file, the other eight carry the style, caption-slot and interval changes the report lists |

Nothing here is cited by the manuscript. The three from 2026-09-21 predate both
the corpus regeneration and the empirical weight-rule port, so their numbers are
from a state of the project that no longer exists.

The `.pdf` of the `WassVsResultDiff` row was removed on 2026-10-07: it was a 66 MB byte copy
of a figure that still exists as `FIG_WassVsResultDiff`, and files over 50 MB
cannot go to GitHub.
