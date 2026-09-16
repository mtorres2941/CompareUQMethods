# Baseline assessments of the code and the analysis

The reference point for any before-and-after comparison of this repository.
These describe the code AS ORIGINALLY WRITTEN BY THE AUTHOR, before any
refactoring, at commit `4a20bdd`.

DO NOT EDIT. The value of a baseline is that it was written before the work
began and was never revised to flatter the result. A later assessment goes in
its own file beside this one.

Recorded 2026-09-11. Moved here 2026-09-15 when the stage handoffs were
deleted; the text is unchanged from the original.

## 1. Code quality baseline

Ratings out of 10, with the evidence each rests on.

| Axis | Score | Principal evidence |
|---|---|---|
| Correctness and numerical care | 6 | Weighted G1/G2 bias corrections verified correct algebraically; `wasserstein2_weighted` quantile-grid integration correct; Kish effective sample size used unprompted. Against: `weighted_quantile` indexes unsorted weights against sorted values; `_royston_pvalue` uses the n>=12 polynomials for the 4<=n<=11 branch; `empirical_metadata` can interpolate into `-inf` |
| Code organization and reuse | 3 | Fitting block duplicated verbatim in NB2 cell 12, NB2 cell 20, NB3 cell 15; `df_metrics` rebuilt three times in NB2 (cells 9, 50, 51); `rankth` defined in all three notebooks; NB3 cells 17 and 18 differ by one string; `datavisualization.py` is 32 lines of which 26 are empty separators; `weighted_distance_norm` and `random_samples` imported everywhere, called nowhere |
| Naming and readability | 6 | `weighted_lognorm_fit`, `wasserstein1_weighted`, `empirical_metadata` are clear; the `shapiro_wilk_weighted` docstring is publication quality with references. Against: `generate_dontread` is a double negative; `hi2`/`hi3`/`lo2` unexplained; `t` shadows the type parameter; `random_irregular_dataset` docstring claims a return type and component types that do not match the code |
| Testing and validation | 2 | No test file anywhere. Inline tripwires exist (`bw_dirichlet` variance check, three assertions at the end of `random_irregular_dataset`), so the instinct is present but lives inside hot loops instead of a suite. `weighted_quantile` survived because nothing ever checked it |
| Randomness and reproducibility | 2 | One seed in the project (`np.random.seed(42)`, NB1 cell 10) and it governs a figure. `seed=0` as a default parameter in `generate_random_numbers` collapses the sample space. Dirichlet weights, combo shuffle, and all Monte Carlo draws use global numpy state. The one file needed to reproduce the study is gitignored |
| Performance awareness | 4 | `tqdm` used throughout, so runtime is watched; stale `# takes ~6 minutes for 15k` comment, now 10x pessimistic. Diagnosis was wrong: cost is dpi=1200 rasterization and scalar `.loc` writes, not the mathematics. `np.array([ele for ele in eccs if ele > 0])` appears six times; `np.array([ele**exp for ele in data])` once |
| Scientific Python idiom | 5 | Competent scipy: `set_bandwidth`, `cumulative_trapezoid`, `argrelextrema`, bounded `minimize` for the weighted MLE. Weak pandas: `df.loc[key] = dict` in loops, object-dtype frames, `pd.Series(index=...)` with no dtype, repeated `.rank()` and `.value_counts()` on frames computable once |
| Version control and project hygiene | 4 | Real commit messages, an explanatory `.gitignore`, `setup.py`, MIT license, a good README. Against: `setup.py` declares zero dependencies, nothing pinned, the `waterweed` kernel no longer exists, 267 MB of PNGs committed including 15 orphans from two dead naming schemes, and `DATA_all.json` is gitignored |
| Statistical implementation judgment | 5 | Knowledge is strong (weighted MLE, Shapiro-Francia, Kish, Royston, effective-n bandwidth). Judgment about deployment is weaker: a 27.5% filter keyed to the data's own metrics and preferentially removing high-`weight_outliers` cases, undiscussed; `logfit_offset = 0.5` undocumented; a continuous modality index labelled "Mode Count"; Shapiro-Wilk and Shapiro-Francia compared as if one column |

**Archetype.** Closest to a self-taught researcher who codes to get results, but
an unusually strong one, deviating far in both directions. Above that profile:
statistical sophistication at domain-scientist level, and packaging instincts
(`src/` module, `setup.py`, explanatory `.gitignore`, usable README) that are
research-software-engineer behaviours. Below it: reproducibility discipline
below even the self-taught median, zero tests in a codebase whose entire output
is numbers, and notebooks that only run in the order they happened to be run.
The summary: a domain scientist's statistical training with a self-taught
coder's software practices. The defects found are not in the mathematics; they
are in indexing, defaults, and state.

**The three habits identified as highest payoff**, in order: (1) one
`Generator`, created once and passed explicitly, never a default seed and never
global `np.random`; (2) a test written the moment a function returns a number
verifiable by hand, moved out of hot loops into a file; (3) one sentence in the
code, next to the line, whenever data is filtered, transformed or offset,
saying what it does to the conclusion.

## 2. Analysis quality baseline

The state of the analysis as a piece of science, before changes.

- **Reproducible from a clean clone:** no. Nothing is seeded, and the one
  required data file is gitignored.
- **Results recoverable without a rerun:** partly. NB1 and NB2 write three
  XLSX tables. NB3, which produces the headline result, writes no table at all.
- **Monte Carlo sample size:** 1,000 draws, against 10,000 as stated in the
  brief and manuscript.
- **Normalization:** by the unweighted mean, against the weighted mean as
  stated in the brief.
- **Synthetic dataset diversity:** reduced by the `seed=0` defect. Shape
  parameters are constants across the study; Gaussian components of equal
  length are exact affine images of one another (Pearson r = 1.0).
- **Coverage claim:** 27.5% of generated datasets discarded by a filter keyed
  to the synthetic metric IQR rather than empirical ranges; effective maximum
  `n` is 749 against a stated 1,000; datasets with high outlier weight
  preferentially removed.
- **Comparison design:** independent sampling across UQ methods, no common
  random numbers, diverging from Marsh et al. (in press) and Henriksson et al.
  (2015).
- **Goodness-of-fit metric:** W1 only. No overlap-area or SMAA robustness
  check (Prado-Lopez et al. 2014).
- **Consistency with the author's own KL1/KL2 papers:** bandwidth rule differs
  and is not yet justified in the manuscript.
- **Undocumented methodological choices:** `logfit_offset = 0.5`; the
  `(1-capecc)` divisor on cap rank frequencies; the `**exp` variance inflation
  in generation.
- **Known incorrect outputs:** `wbeci_mean` and `wbeci_stdev` recorded for only
  one of four datasets per combo (NB3 cell 21).
- **Environment:** unpinned and lost.

## 3. Intended use

The author may request a before-and-after comparison on both 8.1 and 8.2 once
the refactor and the Stage 2 methodological work are complete. Later stages
should record their own assessment against these same axes in their own
handoff, and must not edit this section.

