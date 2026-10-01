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

Nothing here is cited by the manuscript. The three from 2026-09-21 predate both
the corpus regeneration and the empirical weight-rule port, so their numbers are
from a state of the project that no longer exists.
