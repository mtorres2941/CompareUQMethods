# HANDOFF stage-2c - The evaluation target

## 0. STATUS, read this first

**The circularity is removed and the paper's central claim is now CONDITIONED
rather than settled.** Scoring every model against the variable-weighted
empirical CDF of its own training data was doing two things at once: rewarding
the most flexible method for flexibility, and making "variable weighting
improves fit" true by construction. Both are fixed, and both fixes move the
answer.

Four results change what the paper says.

1. **The two arms disagree about the FAMILY out of sample, and the disagreement
   is fully explained.** Against the known parent the KDE beats the lognormal;
   cross-validated on the empirical arm the lognormal beats the KDE. Both
   survive a paired bootstrap. The criterion and the size mix account for the
   sign, a factor of two in magnitude is a genuine corpus-to-arm difference, and
   **what every arm and every criterion agrees on is that the KDE loses at
   n = 10-99 and wins at n >= 1000.** Section 4.3.
2. **"Variable weighting improves fit" was 62 percent definitional on the
   empirical arm.** On a common target it is a coin flip overall and a size
   effect underneath: for the KDE, variable weighting is worse by 0.0395 at
   n = 3-9 and better by 0.0571 at n >= 1000. Section 4.4.
3. **The KDE's advantage is not about multimodality.** Within every size band the
   modality split barely moves the answer and the size split decides it. In the
   visibly unimodal majority the lognormal gets the MEAN slightly better and
   loses on SHAPE by more than a factor of two. Section 4.6.
4. **On real data the three-parameter lognormal is indistinguishable from
   gamma.** Every paired interval straddles zero. It keeps its place because it
   is never worse, but the paper must say so. Section 4.7.

**Nothing here is waiting on a person except three author decisions, all of them
"leave it alone" recommendations**: the guard on the Silverman bandwidth, the
scoring grid's quadrature route, and whether to add a tail-sensitive companion
to W1. Sections 4.8, 4.9 and 4.10.

### What actually needs your eyes

This document carries the findings. Two things it cannot carry, and one it can.

1. **The three author decisions above.** Each has a recommendation and the
   numbers behind it; none blocks the next stage.
2. **The new markdown in notebook 2**, from "The evaluation target" to the end.
   It is roughly eight cells of prose that will ship in the public deposit and
   that describes the method change to an outside reader. The findings in it are
   in section 4 below, but the WORDING is not, and the notebooks are the entry
   point by decision 3.
3. **Notebook 3 does not need reviewing or re-running.** Nothing on its numeric
   path changed, it calls none of the changed interfaces, and
   `TABLE_PLCAResults.csv` is bit-identical; see section 6. The one thing in it
   this stage touches is listed as an open item for Stage 2g in section 5, not
   as a defect.

The three new figures are `FIG_EvaluationTarget`, `FIG_TargetBySize` and
`FIG_Regret`. They are drawn from the tables and were checked rendered.

**The Stage 2b handoff is deleted**, per the standing rule that only the current
stage's handoff is kept. Its findings survive as decisions 49 to 58 in CLAUDE.md
and entries 35 to 52 in the discrepancy file, and the CLAUDE.md decisions that
pointed at its sections now point at those entries instead.

---

## 1. Stage and branch

| | |
|---|---|
| **Stage** | 2c, the evaluation target |
| **Branch** | `stage-2c-target` |
| **Branched from** | `ef84231`, "Restate the coverage entry, which still carried the withdrawn numbers", on `stage-2b-fitting` |
| **Working tree at branch time** | clean; nothing was uncommitted |

Commits, in order:

| commit | what |
|---|---|
| `3388c9d` | Recover the exact parent of every synthetic dataset (NEUTRAL) |
| `65be458` | Score the synthetic arm against its parent, and cross-validate the empirical arm |
| `ad27327` | Reconcile the two arms, and answer the gamma question out of sample |
| (see `git log`) | the notebook section, the figures, and this handoff |

---

## 2. What was asked

Fix the evaluation target. Every model was scored by W1 against the
variable-weighted empirical CDF of the same data it was fitted to, which
structurally favours the KDE and makes the weighting claim close to true by
construction. Score the synthetic arm against the known parent and
cross-validate the empirical arm; report both beside the old scores. Decompose
the error into fit and definitional parts. Report regret rather than only win
rate. Plus five framing tasks carried in from Stage 2b: adopt and re-examine the
guarded Silverman bandwidth, post-stratify every aggregate, state the empirical
headline as a win share, answer the gamma question, and decide whether W1 needs
a tail-sensitive companion.

---

## 3. What was done

### 3.1 The parent had to be recovered before it could be scored against

`parents.json.gz` does not hold enough to rebuild a parent CDF, and CONTEXT.md
said it did. The record stores each component's moment TARGETS, from which
`components.solve_component` recovers its location and scale deterministically,
plus the global shift and the truncation bounds. It does not store the
displacement the overlap solve gave each component: one solved scalar times k
ordinates from the generator's stream, and one recorded overlap value cannot
identify k - 1 displacements.

`corpus.rebuild_parents` replays the generation loop instead. **This is not a
regeneration and does not reopen anything**; the same argument as decision 58
applies. No corpus is written, nothing is redrawn, and the replay is checked
rather than trusted: it refuses unless `genconfig.DEFAULT` still equals the
recorded configuration, compares twelve record fields plus `pi`, `market` and
`mode_counts` per dataset, and compares the replayed values and weights against
`values.parquet` element by element. **All 10,050 datasets of
`corpus_2026-09-15b` replay byte-identically, in 13 minutes.** Cached as
`parents_spec.json.gz` inside the corpus directory. Decision 64, entry 63.

`tests/test_recovery.py` asserts that the displacements are NOT in the record,
so if the record ever gains them the replay can be replaced by a read and that
test deleted with it.

### 3.2 The new machinery

`src/recovery.py`, 26 tests in `tests/test_recovery.py`. W1 against the parent
under either weighting scheme, the tail charge, overlap area, the mean/shape
split, per-split cross-validation with its spread, cross-validated held-out log
density, the fit-versus-definitional decomposition, regret, post-stratification,
win share, and a paired bootstrap.

The recovery and decomposition columns are computed inside
`comparison.score_methods`, from the SAME fit as the in-sample score, rather
than in a second pass. Refitting would have doubled the notebook's cost and
opened the possibility of two passes fitting slightly different objects.

Notebook 2 gained sixteen cells and three figures. Four audit scripts:
`evaluation_target.py` (the whole measurement, with a `report` mode that redraws
from the tables without recomputing), `bandwidth_against_parent.py`,
`family_out_of_sample.py`, `scoring_grid_error.py`.

### 3.3 One error this stage made and caught

The first run scored each method against the parent it estimates -- the sampling
mixture for a uniform-weighted method, the market-weighted mixture for a
variable-weighted one -- and read the result as a comparison of the two
weighting schemes. **It is not one.** The six methods were being scored against
different truths, so `KDE, Variable` appearing to fall from first to third says
nothing about weighting. `w1_market`, which scores all six against the
market-weighted parent, is the comparison that answers the weighting question,
and it says something quite different. Decision 65.

---

## 4. Numbers that moved

Every number below is new rather than moved, except where marked. **No previously
reported number changes except one**, in section 4.11.

### 4.1 What the new target does to the synthetic headline

Mean W1, synthetic arm, 10,000 datasets:

| method | in sample | against the parent | mean rank, in sample | mean rank, parent |
|---|---|---|---|---|
| `KDE, Uniform` | 0.1612 | **0.1228** | 3.811 | **2.029** |
| `Lognormal, Uniform` | 0.1673 | 0.1306 | 4.355 | 3.005 |
| `KDE, Variable` | **0.0776** | 0.1639 | **1.447** | 2.955 |
| `Lognormal, Variable` | 0.0986 | 0.1699 | 2.483 | 3.720 |
| `Normal, Uniform` | 0.2188 | 0.2103 | 5.046 | 4.478 |
| `Normal, Variable` | 0.1723 | 0.2368 | 3.858 | 4.814 |

Win share moves from `KDE, Variable` 0.706 in sample to `KDE, Uniform` 0.515
against the parent. **The ordering of the two WEIGHTINGS inverts and the ordering
of the three FAMILIES does not.** That is the circularity, seen directly: the
in-sample target is the variable-weighted CDF.

### 4.2 The empirical arm, cross-validated

127 of the 147 datasets reach n = 10 and have a cross-validated score, 20 fits
each. Mean W1:

| method | in sample | cross-validated | CV median | CV rank within weighting | CV win share within weighting |
|---|---|---|---|---|---|
| `Lognormal, Uniform` | 0.1990 | **0.2908** | 0.2580 | **1.646** | **0.496** |
| `Lognormal, Variable` | 0.1687 | 0.3088 | 0.2621 | **1.551** | **0.598** |
| `KDE, Uniform` | 0.1806 | 0.3214 | 0.2709 | 1.827 | 0.331 |
| `KDE, Variable` | **0.1471** | 0.3411 | 0.2927 | 2.024 | 0.197 |
| `Normal, Variable` | 0.3619 | 0.4739 | 0.3637 | 2.425 | 0.205 |
| `Normal, Uniform` | 0.3990 | 0.4810 | 0.3484 | 2.528 | 0.173 |

**These are notebook 2's numbers and they are the canonical ones.**
`audits/evaluation_target.py` runs the same computation on an independent split
stream and gets 0.2910, 0.3132, 0.3223, 0.3467, 0.4761, 0.4793. **The third
decimal of a cross-validated mean moves with the splits; the ordering and every
conclusion below do not.** The synthetic numbers are deterministic and agree
exactly between the two.

**The split noise is the same size as the spread between methods.** Median spread
between the best and worst method on a dataset 0.1200; median split-to-split
standard deviation of one method 0.0959. By band: at n = 10-99 the spread is
0.1169 against a split sd of 0.1126, and at n = 100-999 it is 0.2645 against
0.0839. **A per-dataset claim at n = 10-99 is not supportable; the arm-level
aggregate is.**

### 4.3 THE TWO ARMS DISAGREE, and here is the whole of why

Paired bootstrap over datasets, KDE minus lognormal within a weighting scheme.
Positive means the KDE is better.

| | uniform | variable |
|---|---|---|
| synthetic, against the parent | **+0.0078** [+0.0064, +0.0092] | **+0.0060** [+0.0043, +0.0076] |
| empirical, cross-validated | **-0.0306** [-0.0479, -0.0138] | **-0.0323** [-0.0495, -0.0156] |

Both intervals exclude zero, so the disagreement is not noise. The chain below is
one coherent computation from `audits/evaluation_target.py`, which cross-validates
a 2,000-dataset corpus sample that notebook 2 does not; its empirical step reads
-0.0321 against the notebook's -0.0306, which is the split-stream difference of
section 4.2. Removing one difference at a time, uniform weighting:

| step | KDE minus lognormal |
|---|---|
| synthetic, parent, equal allocation | +0.0078 |
| the SAME corpus, cross-validated instead | -0.0034 |
| synthetic, cross-validated AND reweighted to the empirical size mix | -0.0127 |
| empirical, cross-validated, reweighted | -0.0321 |

Variable weighting: +0.0060, -0.0053, -0.0172, -0.0344.

**The criterion accounts for the first flip** -- a cross-validation half measures
the KDE at n/2 and its advantage is a large-n advantage -- **and the size mix
accounts for the second.** A factor of about two in magnitude remains and is a
genuine corpus-to-arm difference.

**What does not depend on any of this is the SHAPE.** KDE minus lognormal by
band, `*` marks a bootstrap interval excluding zero:

| weighting | arm, criterion | n 3-9 | 10-99 | 100-999 | >= 1000 |
|---|---|---|---|---|---|
| Uniform | synthetic, parent | -0.0112* | -0.0155* | +0.0188* | +0.0392* |
| Uniform | synthetic, CV | - | -0.0210* | -0.0009 | +0.0119* |
| Uniform | empirical, CV | - | -0.0625* | +0.0203 | +0.0164* |
| Uniform | empirical, in sample | +0.0354* | -0.0088 | +0.0811* | +0.0281* |
| Variable | synthetic, parent | +0.0107* | -0.0179* | +0.0023* | +0.0289* |
| Variable | synthetic, CV | - | -0.0253* | -0.0080* | +0.0177* |
| Variable | empirical, CV | - | -0.0586* | +0.0053 | +0.0133* |
| Variable | empirical, in sample | +0.0246* | -0.0088 | +0.1050* | +0.0310* |

Decision 66, entry 54. `audits/evaluation_target.py` section 3c reproduces it.

### 4.4 The weighting claim, decomposed and then re-asked on a common target

**Decomposed.** Every model, including the uniform-weighted ones, is scored
against the variable-weighted empirical CDF, so a uniform-weighted model is
charged a distance no estimation method can remove:

| arm | method | total | own scheme | definitional | definitional share |
|---|---|---|---|---|---|
| empirical | `KDE, Uniform` | 0.1806 | **0.1164** | 0.1119 | **61.9 pct** |
| empirical | `Lognormal, Uniform` | 0.1990 | 0.1595 | 0.1119 | 56.2 pct |
| empirical | `Normal, Uniform` | 0.3990 | 0.3801 | 0.1119 | 28.0 pct |
| synthetic | `KDE, Uniform` | 0.1612 | **0.0574** | 0.1373 | **85.2 pct** |
| synthetic | `Lognormal, Uniform` | 0.1673 | 0.0888 | 0.1373 | 82.1 pct |

The definitional term is identical for all three uniform-weighted methods
because it is a property of the weights alone. **Ranking the six against their
OWN weighting scheme reverses the conclusion on both arms**: `KDE, Uniform` goes
from rank 3 to rank 1, `KDE, Variable` from 1 to 2.

**Re-asked on a common target.** Against the market-weighted parent, where all
six estimate the same thing, mean W1 over 10,000 datasets:
`KDE, Variable` 0.1639, `KDE, Uniform` 0.1665, `Lognormal, Uniform` 0.1673,
`Lognormal, Variable` 0.1699, `Normal, Uniform` 0.2293, `Normal, Variable`
0.2368. The two parents are 0.0828 apart on average before any fitting.

Paired, variable minus uniform, positive = variable better:

| family | mean difference | interval | variable wins | distinguishable |
|---|---|---|---|---|
| Normal | -0.0076 | [-0.0107, -0.0045] | 51.1 pct | yes, and it is WORSE |
| Lognormal | -0.0026 | [-0.0054, +0.0002] | 54.6 pct | no |
| KDE | +0.0025 | [-0.0008, +0.0057] | 54.2 pct | no |

**Overall it is a coin flip. By size band, for the KDE, it is not:**

| band | difference | interval | variable wins |
|---|---|---|---|
| n 3-9 | -0.0395 | [-0.0473, -0.0317] | 39.0 pct |
| n 10-99 | -0.0298 | [-0.0375, -0.0224] | 42.8 pct |
| n 100-999 | +0.0223 | [+0.0179, +0.0264] | 59.0 pct |
| n >= 1000 | +0.0571 | [+0.0539, +0.0604] | 75.8 pct |

The lognormal shows the same pattern more strongly, -0.0580 at n = 3-9 to
+0.0451 at n >= 1000, and every band is distinguishable for both families.

Decision 65, entry 55.

### 4.5 Regret

Mean regret, and the tail that reverses two of them.

Synthetic, against the parent, 10,000 datasets:

| method | mean | p50 | p90 | p95 | max | zero share |
|---|---|---|---|---|---|---|
| `KDE, Uniform` | **0.0290** | 0.0000 | 0.0844 | 0.1506 | 1.2635 | **0.516** |
| `Lognormal, Uniform` | 0.0368 | 0.0204 | 0.0874 | **0.1281** | **0.8621** | 0.164 |
| `KDE, Variable` | 0.0702 | 0.0170 | 0.1860 | 0.3142 | 3.7123 | 0.181 |
| `Lognormal, Variable` | 0.0762 | 0.0346 | 0.1838 | 0.2993 | 2.3243 | 0.078 |
| `Normal, Uniform` | 0.1166 | 0.0741 | 0.2796 | 0.3621 | 1.4910 | 0.037 |
| `Normal, Variable` | 0.1431 | 0.0891 | 0.3270 | 0.4551 | 3.3246 | 0.024 |

Empirical, cross-validated, 127 datasets, mean / p95 / zero share:
`Lognormal, Uniform` 0.0244 / 0.0953 / 0.331, `Lognormal, Variable`
0.0425 / 0.1454 / 0.150, `KDE, Uniform` 0.0550 / 0.2411 / 0.268,
`KDE, Variable` 0.0748 / 0.2702 / 0.047, `Normal, Variable`
0.2075 / 0.7192 / 0.055, `Normal, Uniform` 0.2147 / 0.7061 / 0.150.

**The sentence the numbers support is that the KDE has the lower mean cost and
the lognormal the tighter worst case.** Entry 56.

### 4.6 The KDE's advantage is not about multimodality

Two things had to be controlled or the answer is an artifact:
`modality.n_modes_visible` returns 1 below n = 8 without measuring anything, so
the whole of stratum 1 would be counted unimodal by fiat; and multimodality is
only resolvable at large n, where every method does better. Datasets below n = 8
are dropped and the split is reported within each band.

At n >= 8 the corpus is 93.7 percent visibly unimodal and the empirical arm 95.2.
`KDE, Uniform` mean rank against the parent:

| band | visibly unimodal | multimodal |
|---|---|---|
| n 10-99 | 2.55 (2,404 datasets) | 2.58 (96) |
| n 100-999 | 1.57 (2,335) | 1.47 (165) |
| n >= 1000 | 1.23 (2,272) | 1.25 (228) |

**The modality split barely moves it; the size split decides it.** The KDE's win
share against the lognormal at the same weighting, on visibly UNIMODAL datasets
only: 48 percent at n = 10-99, 82 at n = 100-999, 99 at n >= 1000.

**Where the advantage does come from.** Location against shape, variable
weighting, visibly unimodal, n >= 1000: the lognormal gets the MEAN slightly
better, 0.0159 against the KDE's 0.0173, and loses on SHAPE by more than a factor
of two, 0.0569 against 0.0243. **It is a shape advantage -- skewness and tail
behaviour -- and not an ability to represent several modes.** Entry 57.

### 4.7 The gamma question

Paired bootstrap against the three-parameter lognormal. Positive means the
lognormal is better.

| arm, criterion | vs gamma | vs lognormal_2p | vs lognormal_offset | vs normal |
|---|---|---|---|---|
| empirical CV, uniform | +0.0044 [-0.0025, +0.0119] | +0.0042 [-0.0081, +0.0174] | +0.0044 [-0.0015, +0.0090] | +0.2049 [+0.1465, +0.2808]* |
| empirical CV, variable | +0.0008 [-0.0055, +0.0077] | +0.0080 [-0.0061, +0.0215] | -0.0022 [-0.0110, +0.0037] | +0.1637 [+0.1055, +0.2470]* |
| synthetic parent, uniform | +0.0117 [+0.0104, +0.0129]* | +0.0323* | +0.0042* | +0.0758* |
| synthetic parent, variable | +0.0045 [+0.0031, +0.0060]* | +0.0253* | -0.0010 [-0.0023, +0.0003] | +0.0621* |

**On real data, out of sample, the three-parameter lognormal is
indistinguishable from gamma, from the two-parameter lognormal, and from the
Stage 1 offset method.** Only the normal separates. On the synthetic arm it does
separate from gamma, winning 77.4 and 67.5 percent of datasets.

**Stage 2b's claim that gamma beats the lognormal on the GUARD-BOUND datasets is
withdrawn.** On those 65 empirical datasets gamma wins 47.7 percent out of
sample, a coin flip; on the `interior` datasets it wins 61.4 percent, the
opposite direction. Decision 70, entry 59.

### 4.8 The bandwidth against the parent: confirms Scott is wrong, does not confirm the guard

The referee is unbiased, which had to be established before it could be used:
only **1.2 percent** of datasets put the optimal bandwidth at the sweep floor of
0.02 of Scott's, against in-sample W1 minimizing there for 95 of 147. Its optimum
sits at a median of 0.46 (uniform) and 0.56 (variable) of Scott's.

Mean W1 against the parent, 800 corpus datasets:

| rule | uniform | variable | median h / sd, uniform | excess over the optimum |
|---|---|---|---|---|
| parent-optimal | 0.0911 | 0.1201 | 0.1899 | 1.000x |
| pure Silverman | **0.1169** | **0.1593** | 0.2376 | 1.113x |
| guarded Silverman | 0.1180 | 0.1597 | 0.2454 | 1.122x |
| Scott | 0.1281 | 0.1695 | 0.4179 | 1.386x |

The guarded rule beats Scott on 72.1 and 66.4 percent of datasets. **Pure
Silverman beats the guarded rule on 90.1 and 83.2 percent.** The guard costs 0.9
and 0.25 percent of mean W1 and buys the repaired p05 of held-out likelihood it
was chosen for.

**Said and left alone, per the stage instruction.** The reconciliation belongs in
the text: a density criterion and a CDF criterion want different bandwidths,
because the empirical CDF is already root-n consistent so smoothing buys a CDF
criterion very little. **Author decision; Stage 2h owns the sweep.** Decision 71,
entry 60.

### 4.9 The scoring grid, measured

Against a 200,001-point lattice the study's criterion is off by a median of 0.20
percent on the empirical arm and 0.11 on the synthetic, p99 about 14.5 percent on
both, and it picks a different winner on 1.36 and 0.50 percent of datasets. **It
is biased BY METHOD and AGAINST the KDE**: `KDE, Variable`'s empirical mean is
0.1471 against a dense 0.1406, a +4.7 percent bias, against +0.2 percent for both
lognormals.

**It does not reach a conclusion, and that is what decides it.** The paired
cross-validated KDE-minus-lognormal difference is -0.0340 at the study's grid and
route, -0.0341 integrating the same grid as two CDFs, and -0.0339 at 20,000
points. The discretization is common to all six methods and cancels.

One free improvement is **not** taken: on the same 1,000 points the CDF route has
a p99 relative error of 2.5 percent against the atom route's 14.5, at the same
cost. Switching moves every reported number for no change in any conclusion.
Author decision. Decision 72, entry 61.

### 4.10 Does W1 need a tail-sensitive companion? No, as long as the guard holds

Integrating each fitted model's survival function beyond the recovery grid, over
60,000 fits: mean charge 0.0000 to 0.0001 by method, **no fit whose unseen tail
exceeds its body score**, and a rank correlation of 1.0000 between the body score
and the total. Mean rank changes in the fourth decimal.

Stage 2b's pathology -- a model with a standard deviation of 3,281 on data whose
own is 0.6 -- was produced at `PROFILE_DELTA_LO_FRAC = 0.01` and does not occur
at 0.25. **So W1 alone is sufficient as long as the guard holds, and
`model_sd_ratio` stays as the sentinel. Stage 2h sweeps that guard and must
report `model_sd_ratio` with every value it tries.** Decision 69.

### 4.11 THE ONE PREVIOUSLY REPORTED NUMBER THAT MOVES

`genconfig.EMPIRICAL_STRATUM_SHARE` was measured on the 149-dataset arm, before
decision 61 dropped `Chairs` and `Grouting`. Corrected to 20 / 78 / 38 / 8 over
147. It is used by `coverage.post_stratified` in notebook 1, whose
`median_post_stratified` column moves by at most **0.27 percent relative**:

| metric | before | after |
|---|---|---|
| `weight_outliers` | 0.036651 | 0.036550 |
| `entropy` | 2.925022 | 2.918925 |
| `skewness` | 1.153434 | 1.150992 |
| `coeffvar` | 0.508989 | 0.508389 |
| `crit_bw_1` | 0.666286 | 0.666253 |

All nine move in the third or fourth decimal. Nothing else in the study reads
that constant; `recovery.empirical_size_shares` measures the shares from the arm
it is handed, so the score tables cannot inherit a stale one. Entry 62.

### 4.12 Overlap area, the carried-forward robustness check

On the synthetic arm, where a reference density exists, overlap area and W1 pick
the same winner on **66.6 percent** of datasets, correlate at Spearman **0.689**,
and **give the same mean-rank ordering of all six methods**. It is not adopted
because it needs a density and the empirical target is a set of atoms; supplying
one would mean a bin width or a kernel, and a kernel would score the KDE against
a KDE. Decision 69, entry 58.

### 4.13 Two defects found and fixed in passing, both introduced by this stage

**A filename collision.** Notebook 2's new post-stratification table was given
the name notebook 1 already uses for the post-stratified dataset
CHARACTERISTICS. Running the pair would have left one table on disk describing
something other than its name, and nothing would have caught it because each
notebook runs green on its own. Renamed to `TABLE_PostStratifiedScores.csv`, and
`tests/test_notebooks.py::test_no_two_notebooks_write_the_same_output_file` now
guards the class. **Stage 3 owns the duplicate-filename check; this is a partial
down payment on it and does not close it, because it only sees literal paths.**
The rename also had to be made in the FIGURE that reads the table, which it was
not at first, and the symptom was `'DataFrame' object has no attribute 'arm'`
rather than a missing file, because notebook 1's table exists and has different
columns.

**A column name that shadows a DataFrame method.** A summary column called
`rank` makes `frame.rank` the method rather than the column, so
`h.rank.get(...)` returned a bound method and the figure raised
`'function' object has no attribute 'get'` twelve minutes into a run. The same
thing happened in an audit script with a column called `tail`. Renamed to
`mean_rank`, and the surviving attribute access is bracket access with a comment
saying why.

---

## 5. Open questions and flags

### Carried forward

Every still-open item from every earlier handoff, restated. An item leaves this
list only by being marked resolved, with the reason.

| Item | Owner | Status |
|---|---|---|
| Bandwidth rule, KL1/KL2 inconsistency | 2h | **RESOLVED** by decision 54 in favour of KL2's rule, and re-examined in 2c against the parent (section 4.8). What stays open for 2h is the SWEEP, and the GUARD, which the parent referee does not confirm |
| Dependent sampling | 2e | STILL OPEN. `rvs_from_uniform` is in place for it |
| Overlap area alongside W1 | 2c | **RESOLVED in 2c.** Section 4.12, decision 69, entry 58 |
| W1 has no complexity penalty | 2c | **RESOLVED in 2c.** Both arms now have a non-circular criterion; sections 4.1 to 4.3 |
| The W1 grid is linear | 2c | **RESOLVED in 2c.** Measured, does not reach a conclusion, left alone. Section 4.9, decision 72 |
| Shapiro-Wilk vs Shapiro-Francia and `_royston_pvalue` | 2f | STILL OPEN |
| `(1-capecc)` divisor | 2g | STILL OPEN |
| `weighted_quantile` must stay fixed before Silverman | 2h | STILL OPEN as a constraint. Silverman is now the production rule and the fix is in place |
| Deduplicated empirical variant | 2h | STILL OPEN. Primary stays EPD-level uniform |
| `mode_share_alpha` at 10 | 2h | STILL OPEN |
| `trunc_iqr_mult` sweep | 2h | STILL OPEN |
| `PROFILE_DELTA_LO_FRAC` sweep | 2h | STILL OPEN, and **now load-bearing**: section 4.10 says W1 needs no tail companion only because the guard is at 0.25. Any sweep of it must report `model_sd_ratio` |
| Kurtosis undefined in stratum 1 | 2f | STILL OPEN |
| `min_mode_sd_frac = 0.15` has no empirical anchor | 2h | STILL OPEN |
| Six or more modes, 5.6 pct of corpus vs 0.7 empirical | 2h | STILL OPEN |
| `SUPP_DatasetExamplesByStratum.png` x-axis is misleading | 3 | STILL OPEN |
| Git history size | 4 | STILL OPEN, decision 28 stands |
| Coverage claim is false at the top of the coefficient of variation | manuscript | STILL OPEN as a text edit, option A, decision 48, with the corrected numbers of decision 63 |
| A single Dirichlet realization moves per-dataset weighted metrics a long way | 2h | STILL OPEN, and **2c adds to it**: the empirical cross-validation cannot compare weighting schemes at all because the weights are exchangeable (decision 65), so averaging over realizations does not fix that particular limitation, it only shrinks the noise |
| Four EC3 parent categories kept as residual bins | - | OPEN as a stated limitation |
| `CementGrout`, `FlowableFill`, `OilPatch` carry a strength field and are not split | - | OPEN by choice |
| `TABLE_MethodCurves.csv.gz` is 91.7 MB and 21x redundant | 3 | STILL OPEN and **closer to the limit**: Stage 2c adds columns to `TABLE_MethodScores.csv`, not to the curves table, so the curves table itself has not grown. Check its size after any run |
| `src/` docstrings still carry stage language | 3 or 4 | STILL OPEN, and `recovery.py` adds more |
| `Aggregates` and `PowerCabling` remain the arm's dispersion extremes | - | OPEN as decision 48's stated limitation |
| Notebook 2 carries no stored outputs | 3 | STILL OPEN |
| `customstats.weighted_lognorm_fit` is unused by the production path | 3 or 4 | STILL OPEN |
| The ICE database figure in `MASS_ECC_CEILING`'s docstring is unsourced here | manuscript | STILL OPEN. Verify before it goes in the paper |
| The empirical W1 values in the manuscript are in raw category units | manuscript | STILL OPEN. Text must take new numbers from the rerun |
| Calibration gate marginally outside noise | author | STILL OPEN; recommendation remains to do nothing, decisions 47, 48, 55 and 63 |
| One cable record wrong by four orders of magnitude | author or 2h | **RESOLVED** by decision 63 |
| EAF against BOF steel is not available | - | CLOSED as infeasible |
| Whether to filter contaminated categories on metadata | - | **RESOLVED**, decisions 60 and 61 |

### New in Stage 2c, and still open

| Item | Owner | Note |
|---|---|---|
| **The guard on the Silverman bandwidth is not confirmed by the parent referee** | author, then 2h | Pure Silverman beats it on 83 to 90 percent of datasets against the parent, at a cost of 0.9 and 0.25 percent of mean W1. The guard buys the held-out-likelihood tail it was chosen for. **Recommendation: leave it.** Section 4.8 |
| **The scoring grid's quadrature route** | author | The CDF route is strictly better at the same cost, p99 2.5 percent against 14.5, and switching moves every reported number for no change in any conclusion. **Recommendation: leave it.** Section 4.9 |
| **A factor of two in the corpus-to-arm gap is unexplained** | 2f, or nobody | Section 4.3 accounts for the sign of the disagreement and not its size. It could be the corpus's shapes, the corpus's weights, or the empirical arm's small n. **No stage owns it and it may not need one**: the conclusion the paper states, the size dependence, is the same on both arms |
| **The empirical arm cannot answer the weighting question at all** | manuscript | Its weights are a flat Dirichlet stand-in with no market information, so no out-of-sample comparison across weighting schemes is meaningful on it. The weighting claim rests entirely on the synthetic arm's market parent. **This is a limitation the paper must state**, and it is the strongest argument in the project for Marsh, Hattam and Allen (2025)-style real production volumes |
| **Notebook 2's runtime** | 3 | Roughly 35 minutes now: the corpus is fitted once for the recovery columns and the empirical arm is cross-validated at ten repeats. `COMPAREUQ_SMOKE_COMBOS` does not apply to notebook 2 |
| **Notebook 3 cell 45 still explains pLCA outcomes with the IN-SAMPLE score** | 2g | It computes `score_all_models`, the circular in-sample W1, and cell 46 plots the pLCA results against it. Notebook 3 runs on the SYNTHETIC corpus, so `w1_parent` is available for exactly those datasets and is the better explanatory variable. **Not changed here**, because Stage 2g owns the sensitivity of the pLCA metrics and changing it would move a figure this stage was not asked to touch. Cells 37 to 40 are NOT affected: they correlate pLCA differences against the distance between two METHODS' fitted models, which needs no target and is not circular |

---

## 6. Inputs and outputs

**Read:** `CLAUDE.md`, `CONTEXT.md`, `reports/HANDOFF_stage-2b.md`, `src/`,
`notebooks/02`, `data/processed/corpus_2026-09-15b/`,
`data/raw/ec3_raw_ecc_2026-08-14.csv.gz` via `src/empirical.py`.

**Written:** `src/recovery.py`; `tests/test_recovery.py`;
`audits/evaluation_target.py`, `bandwidth_against_parent.py`,
`family_out_of_sample.py`, `scoring_grid_error.py`;
`data/processed/corpus_2026-09-15b/parents_spec.json.gz`;
`reports/HANDOFF_stage-2c.md`. New tables from notebook 2:
`TABLE_TargetComparison.csv`, `TABLE_TargetSummary.csv`,
`TABLE_CrossValidatedScores.csv.gz`, `TABLE_CrossValidatedSummary.csv`,
`TABLE_PairedBootstrap.csv`, `TABLE_WeightingDecomposition.csv`,
`TABLE_WeightingOnCommonTarget.csv`, `TABLE_Regret.csv`,
`TABLE_PostStratifiedScores.csv`, `TABLE_ModalityConditioned.csv`. New figures:
`CompareUQMethods_FIG_EvaluationTarget.png`, `FIG_TargetBySize.png`,
`FIG_Regret.png`.

**Modified:** `src/comparison.py` (the recovery and decomposition columns come
from the same fit), `src/corpus.py` (the replay), `src/mixture.py` and
`src/components.py` and `src/generator.py` (`spec` / `parent_from_spec`),
`src/genconfig.py` (the stratum shares); `notebooks/02`;
`tests/test_notebooks.py`; `CLAUDE.md` (decisions 64 to 72, the roadmap, the
dangling handoff references); `CONTEXT.md`;
`reports/MANUSCRIPT_discrepancies.md` (entries 53 to 63).

**Deleted:** `reports/HANDOFF_stage-2b.md`, per the standing rule that only the
current stage's handoff is kept. Its findings are decisions 49 to 58 and
discrepancy entries 35 to 52; the CLAUDE.md decisions that cited its sections now
cite those entries. Git history retains it.

**Not touched:** the generator's algorithm, `genconfig`'s generation parameters,
the empirical extract, the fitting families, the corpus, and the manuscript.

**NOTEBOOK 3 WAS NOT RE-RUN, and it does not need to be.** Nothing on the numeric
path changed: `fitting.py`, `families.py`, `customstats.py` and `empirical.py`
are untouched in this stage, and the diffs to `components.py`, `generator.py`
and `mixture.py` add a `spec` method and an optional constructor tag and remove
nothing. Notebook 3 calls none of `comparison.score_methods`, `recovery` or
`EMPIRICAL_STRATUM_SHARE`. The regression fixtures pass unchanged. So
`TABLE_PLCAResults.csv` is bit-identical to the Stage 2b run and the pLCA
results in it still stand. **Notebook 1 SHOULD be re-run** before any figure is
taken from it, because `coverage.post_stratified` moves in the fourth decimal;
see section 4.11.

---

## 7. Next stage

**Stage 2d, the flip-probability threshold.** It owns the split of the
uniform-to-variable W1 into location and shape, which Stage 2c deliberately did
not do: 2c decomposed a different quantity, a model's total error into fit and
definitional parts.

Four things 2c leaves it, and 2e and 2g after it.

1. **The target is fixed and the machinery is in `src/recovery.py`.** Anything
   2d, 2e or 2g scores should be scored on a non-circular target: the parent on
   the synthetic arm, cross-validation on the empirical one.
2. **Post-stratify.** `recovery.post_stratify` and `empirical_size_shares`. Every
   headline aggregate from here on is reported both ways, and the corpus's equal
   allocation is a precision choice that flips at least one verdict.
3. **State a win share, not a mean rank, on the empirical arm**, and never make
   a size-banded claim below about n = 100 without the relative-gap view beside
   it. Decision 68.
4. **Never compare weighting schemes out of sample on the empirical arm.**
   Decision 65. The weights there are exchangeable and carry no information that
   generalizes across a split.

**What Stage 2d must NOT do.** Reopen generation; change the fitting families or
the bandwidth, which are decisions 51, 52, 54 and 71; or take the three author
decisions listed in section 5.

### Read this before touching the comparison again

The habits Stage 2b recorded still hold, and this stage adds one.

9. **Ask what two numbers are measuring before you compare them.** The largest
   error in this stage was scoring six methods against the parent each of them
   estimates and reading the result as a comparison of the two weighting schemes.
   The arithmetic was right, every column was correctly computed, and the
   conclusion was meaningless because the six were being judged against different
   truths. Nothing in the code could have caught it; the check is the question.
10. **Never name a column after a DataFrame method.** `rank`, `tail`, `mean`,
    `max`, `count`, `size`, `shape`. Attribute access then silently returns the
    method, and the failure surfaces wherever the value is first used, which in a
    notebook is minutes or tens of minutes later. This stage hit it twice in one
    afternoon. `mean_rank`, not `rank`; and where a column must keep such a name,
    reach it with brackets.
11. **A number quoted in a handoff must come from the table the paper reads.**
    The cross-validated means in section 4.2 were first written from the audit
    script, which uses its own split stream, and differ from notebook 2's in the
    third decimal. No conclusion changed, but the handoff would have quoted
    figures that appear in no committed table. Check the provenance of every
    number before it goes in, and say which artifact it came from.
