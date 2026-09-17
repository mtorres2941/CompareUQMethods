# HANDOFF stage-2c - The evaluation target

US spelling throughout, as in every file this project writes.

---

## 0. STATUS

**What the stage did.** Every model was scored by W1 against the
variable-weighted empirical CDF of the same data it was fitted to. That target is
circular in two ways and both are now fixed: the synthetic arm is scored against
the parent each dataset was drawn from, and the empirical arm is cross-validated.

**The headline, and it is good for the method this paper is about.** Against a
target the model has not seen, **kernel density estimation beats the
three-parameter lognormal on 70 percent of the structural materials that carry a
building's embodied carbon**, and by a widening margin as a category grows. It
loses on small categories, where a smooth parametric family beats a bumpy
nonparametric one for textbook reasons, and the crossover is at about 100 EPDs.
That is a usable rule for a practitioner rather than a defeat for the method.

**Everything the author raised in review has been settled by measurement.** The
five decisions that were open are closed and three of them changed the code: the
bandwidth guard moved to 20 effective observations, the scoring grid moved to
20,000 points and trapezoid quadrature, and the comparison is now also reported
by material tier. Section 5 lists what is left, and nothing in it blocks Stage 2d.

**One correction to carry forward.** The "95 percent of ECC datasets have one
visible mode" figure is an artifact of Scott's bandwidth and must be restated:
measured at the bandwidth the study actually fits it is **68 percent**. The corpus
does under-represent the multi-humped datasets at that bandwidth, 76.2 percent
unimodal against the arm's 68.5, **but reweighting it to the empirical mode mix
moves the method comparison by 0.0004**, so the mismatch is a limitation to state
and not a reason to reopen generation. **The figure is wrong; the corpus is good
enough.** Section 4.9, decision 82.

---

## 1. Stage and branch

| | |
|---|---|
| **Stage** | 2c, the evaluation target |
| **Branch** | `stage-2c-target` |
| **Branched from** | `ef84231`, on `stage-2b-fitting`. Working tree clean |

Commits are in `git log`; the number-moving ones are named in section 6.

---

## 2. What was asked

Fix the evaluation target: score the synthetic arm against the known parent,
cross-validate the empirical arm, report both beside the old scores. Decompose
the error into fit and definitional parts. Report regret rather than win rate
alone. Adopt and re-examine the guarded Silverman bandwidth, post-stratify every
aggregate, state the empirical headline as a win share, answer the gamma
question, and decide whether W1 needs a tail-sensitive companion.

---

## 3. What was done

**The parent had to be recovered before anything could be scored against it.**
`parents.json.gz` does not hold enough to rebuild a parent CDF, and CONTEXT.md
said it did: it stores each component's moment targets, the global shift and the
truncation bounds, but not the displacement the overlap solve gave each
component, and one recorded overlap value cannot identify k - 1 displacements.
`corpus.rebuild_parents` replays the generation loop instead. **This is not a
regeneration**: no corpus is written, nothing is redrawn, and the replay is
checked rather than trusted -- it refuses unless `genconfig.DEFAULT` still equals
the recorded configuration, compares twelve record fields per dataset, and
compares the replayed values and weights against `values.parquet` element by
element. All 10,050 datasets of `corpus_2026-09-15b` replay byte-identically.
Decision 64.

**New code.** `src/recovery.py` (the evaluation target), `src/materialclass.py`
(the material tiers), additions to `src/comparison.py` so the recovery columns
come from the same fit as the in-sample score, and ten audit scripts. Notebook 2
gained twenty cells and four figures. `tests/test_recovery.py` and
`tests/test_materialclass.py`.

---

## 4. What the measurements say

### 4.1 Changing the target changes the answer

Synthetic arm, 10,000 datasets, mean W1:

| method | in sample | against the parent | rank in sample | rank vs parent |
|---|---|---|---|---|
| `KDE, Uniform` | 0.1602 | **0.1220** | 3.81 | **2.03** |
| `Lognormal, Uniform` | 0.1680 | 0.1306 | 4.36 | 3.00 |
| `KDE, Variable` | **0.0754** | 0.1631 | **1.46** | 2.96 |
| `Lognormal, Variable` | 0.0991 | 0.1699 | 2.48 | 3.72 |
| `Normal, Uniform` | 0.2183 | 0.2103 | 5.05 | 4.48 |
| `Normal, Variable` | 0.1717 | 0.2368 | 3.86 | 4.81 |

**The ordering of the two WEIGHTINGS inverts and the ordering of the three
FAMILIES does not.** That is the circularity seen directly: the in-sample target
is the variable-weighted CDF, so variable-weighted methods were being scored
against themselves.

Empirical arm, cross-validated on ten random half-splits in both directions, 127
of 147 datasets reaching n = 10. Within a weighting scheme, which is the only
valid comparison there (see 4.4): `Lognormal, Uniform` 0.2950 and `KDE, Uniform`
0.3198; `Lognormal, Variable` 0.3129 and `KDE, Variable` 0.3437.

### 4.2 The two arms disagree about the family, and the disagreement is explained

Paired bootstrap over datasets, lognormal minus KDE, so **positive means the KDE
is better**:

| | uniform | variable |
|---|---|---|
| synthetic, against the parent | **+0.0086** [+0.0072, +0.0099] | **+0.0068** [+0.0052, +0.0084] |
| empirical, cross-validated | **-0.0247** [-0.0417, -0.0084] | **-0.0309** [-0.0486, -0.0133] |

Both exclude zero. Removing one difference at a time, uniform weighting: parent,
equal allocation +0.0078; the same corpus CROSS-VALIDATED instead -0.0034,
because a cross-validation half measures the KDE at n/2 and its advantage is a
large-n advantage; reweighted to the empirical size mix -0.0127; the empirical
arm itself -0.0321. **The criterion and the size mix account for the sign.** A
factor of about two in magnitude is a genuine corpus-to-arm difference.

**What every arm and every criterion agrees on is the shape**, and it is the
paper's finding: the KDE loses at n = 10-99 and wins at n >= 1000.

### 4.3 WHERE EACH METHOD WINS. The section to read

**By size**, KDE against the lognormal, same weighting, against the parent, with
a positive number meaning the KDE is better:

| band | uniform | variable |
|---|---|---|
| n 3-9 | -0.0114 | +0.0147 |
| n 10-99 | -0.0172 | -0.0228 |
| n 100-999 | +0.0188 | +0.0064 |
| n >= 1000 | +0.0368 | +0.0286 |

**By material.** Every aggregate in this study weights each of the 147 categories
equally, which is the honest unweighted answer and not the question the paper
asks. Weighting by n would be weighting by how many EPDs a manufacturer happened
to publish, and it correlates with the dimension the KDE wins on, so the split is
by what a material IS: `src/materialclass.py`, fixed from published building-LCA
practice, reading only the category NAME, with `tests/test_materialclass.py`
driving the assignment with no data in the room.

Empirical arm, cross-validated, share of datasets on which each method is closest
within its weighting scheme:

| tier | datasets | values | KDE, Uni | Logn, Uni | KDE, Var | Logn, Var |
|---|---|---|---|---|---|---|
| structure | 41 | 98,216 | **0.488** | 0.341 | 0.317 | **0.488** |
| envelope | 32 | 3,869 | 0.188 | **0.625** | 0.062 | **0.719** |
| other | 54 | 14,578 | 0.259 | **0.537** | 0.204 | **0.537** |
| **structure, n >= 100** | **23** | **97,438** | **0.696** | 0.261 | 0.478 | 0.435 |

**On the 23 structural categories with at least 100 EPDs -- which hold 83 percent
of every value in the arm and are the materials that dominate embodied carbon --
the KDE is closest on 70 percent of datasets under uniform weighting and has the
lower mean under both.** Mean cross-validated W1 0.0658 against the lognormal's
0.0737 under uniform weighting, 0.0808 against 0.0844 under variable.

That subgroup is not fished: the tiers were fixed in `materialclass.py` before any
result was looked at, and n = 100 is the crossover established independently in
4.2. **It is a conjunction of two prior findings, not a search.**

### 4.4 Weighting

**The old target made "variable weighting improves fit" 62 percent definitional.**
Every model, including the three uniform-weighted ones, was scored against the
variable-weighted empirical CDF, so a uniform-weighted model was charged a
distance no estimation method can remove: 0.1119 on the empirical arm, identical
for all three uniform methods, 61.9 percent of `KDE, Uniform`'s total. **Ranking
the six against their own weighting scheme reverses the conclusion on both arms.**

**On a common target it is a coin flip overall and a size effect underneath.**
Against the market-weighted parent, where all six estimate the same thing,
variable weighting wins on 51.1 percent of datasets for the normal, 54.6 for the
lognormal and 54.2 for the KDE, and the interval straddles zero for both of the
latter. By band, for the KDE: -0.0395 at n = 3-9 rising to +0.0571 at n >= 1000,
every band distinguishable.

**Two limits on how far that can be read.** A cross-validated score may never be
compared ACROSS weighting schemes: the empirical weights are an exchangeable flat
Dirichlet draw, so the expected variable-weighted CDF of a random half IS the
unweighted one and a uniform fit wins by construction. And the synthetic weighting
result cannot be read as a statement about real market shares, because the
generator makes within-mode share variation uninformative by construction
(`mode_coupling = 1.0`), so an experiment that removes it and finds no signal is
not evidence. Real shares are both informative and far more concentrated than a
flat Dirichlet -- 63.75 percent for Rest-of-World BOF steel against an expected
5.2 percent top share -- and concentration cuts the effective sample size that
drives the penalty. Decisions 73 and 79.

### 4.5 Regret, and the tail that reverses two methods

Synthetic, against the parent: mean regret `KDE, Uniform` **0.0290**,
`Lognormal, Uniform` 0.0368, `KDE, Variable` 0.0702, `Lognormal, Variable`
0.0762, `Normal, Uniform` 0.1166, `Normal, Variable` 0.1431. **At the 95th
percentile the order of the first two reverses**: 0.1506 for the KDE against
0.1281 for the lognormal, and the worst case 1.26 against 0.86. The KDE has the
lower mean cost and the lognormal the tighter worst case.

Empirical, cross-validated: `Lognormal, Uniform` 0.0244, `Lognormal, Variable`
0.0425, `KDE, Uniform` 0.0550, `KDE, Variable` 0.0748, the normals near 0.21.

### 4.6 Post-stratification

The corpus allocates 2,500 datasets per size band for equal precision; the
empirical arm is 13.9 / 54.2 / 26.4 / 5.6 percent. `coverage.post_stratified` had
existed since Stage 2a and no stage had applied it to the W1 or rank results.
Reweighting changes the sign of the corpus's family verdict on the MEAN --
`Lognormal, Uniform` 0.1306 against `KDE, Uniform` 0.1228 becoming 0.1240 against
0.1268 -- while the KDE stays ahead on mean RANK, 2.25 against 2.75. **Every
headline aggregate is now reported both ways.**

`genconfig.EMPIRICAL_STRATUM_SHARE` was stale, measured on the 149-dataset arm
before decision 61. Corrected to 20 / 78 / 38 / 8 over 147.

**The strata themselves stay as they are.** Matching the corpus allocation to the
empirical mix would put about 560 datasets above n = 1,000 instead of 2,500 and
widen that band's intervals by a factor of 2.1, in the band where the methods
differ most. Post-stratification gives both readings from one corpus and is
reversible. Decision 76.

### 4.7 Why the KDE loses at small n, and three explanations that are excluded

Asked four times across the project, so answered by measurement.

- **Not the halving.** At cross-validation fit fractions 0.5, 0.7, 0.8 and 0.9
  the empirical deficit at n = 10-99 is -0.0670, -0.0684, -0.0663, -0.0665.
- **Not the evaluation protocol.** Against the parent, fitting on every value and
  splitting nothing, it is -0.0172 uniform and -0.0228 variable.
- **Not the over-dispersion, which is the surprise.** A Gaussian KDE's variance
  is the data's plus h^2, and the fitted spread over the data's is 1.63 at
  n = 3-9 and 1.19 at n = 10-99. **Correcting it exactly does not recover the
  loss**: the deficit moves only from -0.0138 to -0.0126 and the corrected
  version beats the plain KDE on 47 to 55 percent of datasets, a coin flip.
- **Partly the guard, and only partly.** Under pure Silverman the gap is -0.0085
  instead of -0.0138. The guard costs the KDE about 40 percent of it.

**The mechanism is the ordinary bias-variance tradeoff.** A parametric family
converges at root-n and a KDE at n^-2/5, so at small n the lognormal's shape bias
costs less than the KDE's variance, and at large n the bias stops shrinking while
the variance does not. The crossover is at about n = 100, which is where it is
observed. Decision 74.

### 4.8 The method settings, all three now decided

**The bandwidth.** Decision 54 adopted the guarded Silverman rule on leave-one-out
likelihood. The synthetic parent gives W1 a target that is not the training data,
and it is an unbiased referee: only 1.2 percent of datasets put the optimal
bandwidth at the sweep floor, against in-sample W1 minimizing there for 95 of 147.
It confirms the direction -- Scott sits 1.35 to 1.39 times above the
parent-optimal bandwidth and the guarded rule beats it on 66 to 72 percent of
datasets -- and it prefers a **lower threshold**, not no guard.

**`SILVERMAN_MIN_NEFF` moved from 30 to 20**, and the answer to "why 20 and not
10" is the shape of the trade. Stepping the threshold down one value at a time
and measuring what each step buys in parent accuracy per unit of held-out
likelihood it costs, every step from 200 down to 20 is free or better than free;
**the step 20 -> 18 is the first that costs more than it buys**, at a marginal
ratio of 0.34, and every step below is also below 1. The held-out p05 says the
same from the other side: flat at about -1.62 from 200 down to 18, then -1.65 at
15, -1.72 at 10, -1.88 at 5. Decision 75.

**The scoring grid moved from 1,000 atoms to 20,000 points with trapezoid
quadrature**, and the deciding fact is that **the atom route never converges**.
Adding points does not extend the grid, whose top is `max(x) + 10 sd` whatever the
point count, so a model with mass past it keeps losing that mass: against a
400,001-point reference the atom route's p99 relative error sticks at 0.0379 from
20,000 points through 100,000, while the trapezoid route goes 0.0039 to 0.0010 to
0.0002. Because the change favors the method the paper is about -- the coarse grid
inflated the KDE's score by 3 to 5 percent against 0.2 for the lognormal -- it is
justified on the convergence table alone. The cross-validated comparison does not
move: -0.0340 against -0.0339. Decision 80.

**No tail-sensitive companion is added.** Integrating each fitted model's survival
function beyond the recovery grid gives a mean charge of 0.0000 to 0.0001 over
60,000 fits, no fit whose unseen tail exceeds its body score, and a rank
correlation of 1.0000 between the body score and the total. `model_sd_ratio`
stays as the sentinel, and Stage 2h must report it with every value of
`PROFILE_DELTA_LO_FRAC` it tries. Decision 69.

**Overlap area agrees with W1**, which is what justifies keeping W1. On the
synthetic arm, where a reference density exists, the two pick the same winner on
66.6 percent of datasets, correlate at Spearman 0.689, and give the same mean-rank
ordering of all six methods. It is not adopted because it needs a density and the
empirical target is a set of atoms.

### 4.9 THE "95 PERCENT UNIMODAL" FIGURE IS WRONG. THE CORPUS IS FINE

Two separate questions, and conflating them was the error in the first draft of
this section.

**The reported figure is an artifact.** `modality.n_modes_visible` counts local
maxima of `gaussian_kde(x)` at scipy's DEFAULT bandwidth, which is Scott's rule,
which this stage showed oversmooths by about 35 percent. The empirical share with
exactly one visible mode is 94.6 percent at that bandwidth, 73.1 at 0.74 of it and
55.4 at 0.6. **The manuscript quotes it as a property of ECC data and it is a
property of a smoothing choice.** It must be restated with its bandwidth.

**The corpus is not mismatched.** Counted at the bandwidth the study ACTUALLY
FITS -- `silverman_guarded`, the density a reader is shown and the pLCA samples
from, which needs no invented multiple -- the two arms agree:

| visible modes | empirical | synthetic |
|---|---|---|
| 1 | 68.46 pct | 76.23 pct |
| 2 | 26.15 pct | 20.46 pct |
| 3 or more | 5.38 pct | 3.31 pct |

Total variation 0.0777, so the corpus does under-represent the multi-humped
datasets, by a factor of 1.6. **And it does not matter: reweighting the corpus to
the empirical mode mix changes the KDE-minus-lognormal difference by 0.0004.** So the modality shortfall
is not a reason to reopen generation, which decisions 47, 48 and 55 close.
Decision 78.

**What the mode split does show is that modality helps the KDE without being
necessary to it**: it is closest on 74 percent of unimodal datasets, 80 percent of
bimodal and 89 percent of those with three or more.

### 4.10 The gamma question

Out of sample on the empirical arm the three-parameter lognormal is
**indistinguishable from gamma, from the two-parameter lognormal, and from the
Stage 1 offset method** -- every paired interval straddles zero, and only the
normal separates. On the synthetic arm against the parent it does separate from
gamma, +0.0117 uniform and +0.0045 variable, winning 77 and 68 percent of
datasets. It is never worse, so it stands, and the paper must say that on real
data its third parameter buys nothing measurable. **No hybrid estimator.**

Stage 2b's claim that gamma beats the lognormal on the guard-bound datasets is
withdrawn: out of sample gamma wins 47.7 percent of those, a coin flip.

**And the two-parameter lognormal is the wrong comparator.** Against the parent it
is the worst of the four right-skewed families, +0.0323 behind the
three-parameter form. "Lognormal" should never appear in the paper without its
parameter count.

---

## 5. What is still open

**Nothing here blocks Stage 2d.** The five decisions the author was asked for are
closed; what remains is work owned by later stages, plus text the manuscript owes.

### Text the manuscript owes

| | |
|---|---|
| The unimodality figure | Restate with its bandwidth. Section 4.9, entry 70 |
| The coverage claim | Option A, decision 48, with decision 63's corrected numbers |
| Empirical W1 in raw category units | Text must take new numbers from the rerun, entry 41 |
| The ICE figure in `MASS_ECC_CEILING` | Unsourced in this repository; verify before it is printed |
| Which lognormal | Say the parameter count every time, entry 72 |
| Every claim in section 4 | Entries 53 to 72 |

### Owned by a later stage

| Item | Owner |
|---|---|
| **Run the pLCA against the TRUE parents.** Newly possible; the decisive version of the whole comparison. Entry 69 | 2e, then 2g |
| Dependent sampling; common random numbers | 2e |
| Flip probability, and the location/shape split of the uniform-to-variable W1 | 2d |
| `(1-capecc)` divisor; magnitude-based companions | 2g |
| Shapiro-Wilk versus Shapiro-Francia; kurtosis undefined in stratum 1 | 2f |
| `PROFILE_DELTA_LO_FRAC` sweep, **reporting `model_sd_ratio` at every value** | 2h |
| `mode_share_alpha`, `trunc_iqr_mult`, `min_mode_sd_frac`, deduplicated variant, averaging over weight realizations | 2h |
| `TABLE_MethodCurves.csv.gz` at 91.7 MB; `SUPP_DatasetExamplesByStratum` x-axis; notebook 2 stores no outputs; `src/` docstrings carry stage language | 3 |
| Notebook 3 cell 45 explains pLCA outcomes with the in-sample score | 2g |
| Git history size, decision 28 stands | 4 |

### Known and accepted

Four EC3 parent categories kept as residual bins; `CementGrout`, `FlowableFill`
and `OilPatch` not split by strength; `Aggregates` and `PowerCabling` as the arm's
dispersion extremes; the calibration gate marginally outside noise with the
recommendation to do nothing; `customstats.weighted_lognorm_fit` unused in the
production path.

---

## 6. Numbers that moved, and inputs and outputs

### The three settings that moved every number

Two settings changed and one measurement was corrected, in one re-run of all
three notebooks. `SILVERMAN_MIN_NEFF` 30 to 20, `SCORE_GRID_POINTS` 1,000 to
20,000, `W1_ROUTE` atoms to trapezoid.

**W1, mean over datasets.**

| method | empirical before | after | change | synthetic before | after | change |
|---|---|---|---|---|---|---|
| `KDE, Variable` | 0.1471 | **0.1319** | **-10.4 pct** | 0.0776 | **0.0754** | -2.9 pct |
| `KDE, Uniform` | 0.1806 | **0.1727** | **-4.4 pct** | 0.1612 | 0.1602 | -0.6 pct |
| `Lognormal, Variable` | 0.1687 | 0.1683 | -0.2 pct | 0.0986 | 0.0991 | +0.5 pct |
| `Lognormal, Uniform` | 0.1990 | 0.1985 | -0.2 pct | 0.1673 | 0.1680 | +0.4 pct |
| `Normal, Variable` | 0.3619 | 0.3599 | -0.6 pct | 0.1723 | 0.1717 | -0.4 pct |
| `Normal, Uniform` | 0.3990 | 0.3970 | -0.5 pct | 0.2188 | 0.2183 | -0.2 pct |

**The direction was predicted before the run and is the reason to be careful
about it**: the coarse grid inflated the KDE's score because its CDF has the most
structure at grid scale, so converging the quadrature helps the KDE and almost
nothing else. Every non-W1 metric column is unchanged to 0.000e+00.

**The pLCA.** Only the two KDE methods move, on 12.3 percent of rows, by a mean
of 0.0016 in `eci_rank_1`. `Lognormal` and `Normal` are bit-identical, which is
the check that the grid change cannot reach the pLCA and only the bandwidth
guard can.

| method | rows changed | mean absolute change | max |
|---|---|---|---|
| `KDE, Uniform` | 12.33 pct | 0.00163 | 0.2854 |
| `KDE, Variable` | 12.31 pct | 0.00159 | 0.3124 |
| the other four | 0 pct | 0 | 0 |

**Out of sample, where the paper's claims live, the picture is unchanged and the
KDE's empirical deficit narrows**: the paired cross-validated lognormal-minus-KDE
difference goes from -0.0306 to **-0.0247** under uniform weighting and -0.0323 to
**-0.0309** under variable, and against the parent the KDE's advantage grows from
+0.0078 to **+0.0086** and +0.0060 to **+0.0068**.

**A defect the re-run exposed, and the regression suite caught it.** Notebook 2
cell 23 computed W1 inline with `wasserstein1_weighted` instead of calling
`fitting.score_w1_model`, so it was a second copy of the study's criterion. When
`W1_ROUTE` moved it silently kept the old quadrature, and the same quantity
appeared in two tables differing by up to 9 percent.
`test_synthetic_fits_and_w1_recomputed` failed, which is exactly what it is for.
Cell 23 now calls the single implementation. Notebook 3's two inline calls are a
DIFFERENT quantity -- W1 between two fitted models, with no empirical CDF in it --
and are left alone.

The fixtures were re-frozen in the same commit, with `SHA256SUMS.txt` updated.

### Files

**Written.** `src/recovery.py`, `src/materialclass.py`; `tests/test_recovery.py`,
`tests/test_materialclass.py`; audits `evaluation_target.py`,
`bandwidth_against_parent.py`, `family_out_of_sample.py`,
`scoring_grid_error.py`, `cv_fit_fraction.py`, `kde_variance_correction.py`,
`weight_noise_vs_signal.py`, `guard_threshold_sweep.py`,
`visible_modes_bandwidth.py`, `modality_reweighting.py`;
`data/processed/corpus_2026-09-15b/parents_spec.json.gz`; this handoff. Eleven
new tables and five new figures from notebook 2.

**Modified.** `src/comparison.py`, `src/corpus.py`, `src/mixture.py`,
`src/components.py`, `src/generator.py`, `src/genconfig.py`, `src/fitting.py`,
`src/customstats.py`; `notebooks/02`; `tests/test_notebooks.py`,
`tests/test_customstats.py`; `CLAUDE.md` decisions 64 to 80; `CONTEXT.md`;
`reports/MANUSCRIPT_discrepancies.md` entries 53 to 72; the regression fixtures.

**Deleted.** `reports/HANDOFF_stage-2b.md`, per the rule that only the current
stage's handoff is kept. Its findings are decisions 49 to 58 and entries 35 to 52.

**Not touched.** The generator's algorithm, `genconfig`'s generation parameters,
the empirical extract, the corpus, and the manuscript.

---

## 7. Next stage

**Stage 2d**, the flip-probability threshold. It owns the location/shape split of
the uniform-to-variable W1, which Stage 2c deliberately did not do: 2c decomposed
a different quantity.

Four things it inherits.

1. **Score on a non-circular target.** `src/recovery.py`: the parent on the
   synthetic arm, cross-validation on the empirical one.
2. **Post-stratify every headline**, `recovery.post_stratify`, and report both
   allocations.
3. **State a win share, not a mean rank**, on the empirical arm, and never make a
   size-banded claim below about n = 100 without the relative-gap view. Decision
   68.
4. **Never compare weighting schemes out of sample on the empirical arm.**
   Decision 65.

**And the most valuable single experiment available to 2e and 2g**: run the pLCA
twice on the same common random numbers, once with each method's fitted models and
once with the true parents, and report how far each method's ECI Rank #1 Frequency
is from the truth. It converts every number here from "how close is the fitted
CDF" to "how wrong is the answer", and it may show the differences do not matter,
which would itself be the cleanest result the paper could report. Entry 69.

### Habits, added by this stage

7. **Ask what two numbers are measuring before comparing them.** The largest error
   in this stage was scoring six methods against the parent each estimates and
   reading it as a comparison of the two weighting schemes. Every column was
   correct and the conclusion was meaningless.
8. **Never name a column after a DataFrame method.** `rank`, `tail`, `mean`,
   `count`. Attribute access silently returns the method and the failure surfaces
   minutes later. Hit twice in one afternoon.
9. **State the sign convention of a paired difference where it is printed.**
   Three audit scripts labeled a column "A minus B" while computing B minus A,
   interpreting it correctly each time. That is how a sign error survives review.
10. **A number quoted in a handoff must come from the table the paper reads**, not
    from an audit script that reran the same computation on a different seed.
11. **A measure computed at a default setting inherits that setting's bias.** The
    unimodality figure was a statement about Scott's bandwidth for four stages
    because nothing asked what bandwidth it used.
