# Regression fixtures

Reference artifacts for the Stage 1 refactor. They exist so that a phase
labeled NEUTRAL can be proven not to have changed any result. They are not a
claim that these values are methodologically correct: several known defects
are still present in the code that produced them, and Stage 2 will deliberately
change many of these numbers.

## Provenance

These three tables were copied byte for byte from `outputs/tables/` at commit
`de26b4f`, before any Stage 1 change and before any notebook was re-executed.
They are the tables that the current manuscript draft reports. The environment
that originally produced them (Jupyter kernel `waterweed`, Python 3.11) no
longer exists, so they cannot be traced to a pinned software stack.

| File | Produced by | Deterministic? |
|---|---|---|
| `TABLE_EmpiricalECCMetrics.xlsx` | NB1 cell 12 | Yes, given `dct_realeccs_trimmed.json` |
| `TABLE_EmpiricalECCMetricsAndW1.xlsx` | NB2 cell 22 | Yes, given the above and the fitting code |
| `TABLE_SyntheticECCMetricsAndW1.xlsx` | NB2 cell 32 | Yes, given `DATA_all.json` |

All three are deterministic because neither NB1's metric computation nor NB2's
fitting and scoring draws a random number. The randomness in NB1 affects only
dataset generation, which is not rerun (`generate_dontread = False`), and the
randomness in NB2 affects only two illustrative figures.

`SHA256SUMS.txt` records the checksums as frozen.

## Re-freezings

A fixture is re-frozen in the same commit that moves the number, with the delta
in the commit message, per CONTEXT.md section 7.

**Stage 2f, 2026-09-19.** `fit_norm_SW` and `fit_lognorm_SW` became
`fit_norm_SF` and `fit_lognorm_SF`: the study's normality statistic is
Shapiro-Francia under BOTH weightings, where the uniform column used to be the
true Shapiro-Wilk. **Only the two `_uw` columns moved** -- 141 of 147 empirical
rows to a maximum of 0.0266, and 9,388 of 10,000 synthetic rows to a maximum of
0.0358. No W1 column and no other characteristic moved by more than 1e-9, which
is the check that the change reaches only what it should.

**Stage 2f review, 2026-09-21.** `TABLE_EmpiricalECCMetrics.xlsx` gains
`modality_index_fitted` and `modality_index_fitted_uw`: the author's modality
index computed at the bandwidth the study FITS, beside the existing column,
which is the same index at Scott's rule and is untouched. **Two columns added,
nothing moved** -- every shared column agrees to better than 1e-9 on all three
fixtures. The bandwidth is what the measure had been missing: at Scott's rule
it ranks 21st of 23 characteristics for predicting which method fits better,
and at the fitted bandwidth 2nd.

## What is deliberately absent

There is no pLCA fixture here. NB3 writes no table at all, and its Monte Carlo
draws are unseeded, so at this point its results are neither persisted nor
reproducible. Phase 1 persists them as an archive; a genuine pLCA regression
fixture becomes possible only after Phase 2 establishes seeding. See
`tests/fixtures/plca/README.md` once that exists.


---

## Re-frozen in Stage 2h, 2026-09-25

The three tables were re-frozen from `outputs/tables/` after the empirical arm
changed its market-share weight rule. **Re-freezing removes the tripwire for
that one transition, so what moved is recorded here rather than reset
silently.**

### Why they moved

`src/empirical.py` now draws market shares with `weighting.coherent_weights`
at `WEIGHT_RHO = 0.5` instead of a flat Dirichlet over every declaration, so
both halves of the study attach share to a group of adjacent coefficients and
split it inside. The corpus was NOT regenerated; the active corpus is
unchanged.

### What moved, and the control that says it is the weights

| table | moved | did not move |
|---|---|---|
| `TABLE_EmpiricalECCMetrics.xlsx` | all 12 weighted characteristics | all 12 UNWEIGHTED twins, bit for bit |
| `TABLE_EmpiricalECCMetricsAndW1.xlsx` | the same, plus all six W1 scores | 11 of 28, all unweighted |
| `TABLE_SyntheticECCMetricsAndW1.xlsx` | **nothing** | every shared column, to 1e-12 |

Median absolute change on the empirical arm: `coeffvar` 0.066,
`w_v_uw_wasserstein` 0.066, `entropy` 0.135, `skewness` 0.648.

**ALL SIX W1 SCORES MOVED, INCLUDING THE THREE UNIFORM-WEIGHTED ONES, and that
is correct rather than surprising.** Every model in this study is scored
against the VARIABLE-weighted empirical CDF, including the uniform-weighted
fits, so changing the weights changes the TARGET for all six.

The synthetic table failed only because it GAINED two columns,
`modality_index_fitted` and `modality_index_fitted_uw`. No shared value moved
by more than 1e-12. That column is the modality measure decision 134 settled
on and the Stage 2g handoff recorded as owed to whichever stage next reran the
second notebook.

## Re-frozen in Stage 2h, 2026-09-25, for the regenerated corpus

`TABLE_SyntheticECCMetricsAndW1.xlsx` only. **The two EMPIRICAL fixtures did
NOT move and were not touched**, which is the control: regenerating the
synthetic corpus must not change the real categories, and it did not.

The corpus was regenerated as `corpus_2026-09-25` with `min_q1_over_iqr` 0.5 to
0.2 and `cv_log10_mean` 0.129 to 0.329, to widen its dispersion (decision 193).
Every dataset is a fresh draw, so this fixture cannot be compared row by row --
row 7 of the old file and row 7 of the new one are different datasets. What is
checked instead is that the DISTRIBUTIONS moved the way the change predicts and
that the invariants held exactly.

**The invariants, which are what make this a re-freeze rather than a guess:**

    mean_uw                  1.0000 -> 1.0000   exactly, by construction
    n, median                  99.5 ->   99.5   the strata are unchanged
    k, median                     3 ->      3
    is_probe, n_components_dropped        unchanged

**The intended moves, against the real categories they are matched to:**

    coeffvar            0.5032 -> 0.6148   real 0.6406   now much closer
    coeffvar_uw         0.5499 -> 0.6612   real 0.6785
    w_v_uw_wasserstein  0.0941 -> 0.1164   real 0.1475   closer
    skewness            1.3397 -> 1.5944   real 1.7773   closer
    kurtosis            4.2177 -> 5.2588   real 4.6616

**The consequential moves, all the same absolute-scale effect.** Every one of
the six W1 columns rises by 16 to 45 percent, because a more dispersed dataset
has a wider parent and a larger absolute distance to it. Divided by each
dataset's own spread the distance to the truth is 0.965 times the shipped
corpus's, i.e. slightly BETTER. Do not read the raw rise as a worse fit.

    KDE, Variable       0.0364 -> 0.0527
    KDE, Uniform        0.1129 -> 0.1408
    Lognormal, Uniform  0.1214 -> 0.1526
    Lognormal, Variable 0.0714 -> 0.0830
    Normal, Uniform     0.1816 -> 0.2415
    Normal, Variable    0.1340 -> 0.1789

**One that looks backwards and is not.** `truncated_mass` rises 0.0961 to
0.1622 although the truncation bounds are WIDER, 216 against 27 as a multiple.
The parents are far more dispersed, so there is more tail mass to lose even
through wider bounds.

Verified before re-freezing: 588 of 591 tests passed, and all three failures
were this fixture and the two tests that read it.
