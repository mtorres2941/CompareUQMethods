# The generator-parameter sweeps, Stage 2h

These reuse `audits/tune_configuration.py`, which already samples a candidate
configuration, compares every characteristic's DISTRIBUTION against the
empirical arm, and returns the objective. Nothing new was written for them,
because a second implementation of the objective is a second thing that can
drift from the first.

Each command writes `outputs/tables/audits/TABLE_TuningSweep.csv`, so run them
one at a time and rename the output, or run the combined spec at the bottom.

**None of these regenerates the corpus.** `tune_configuration` samples
datasets under a candidate configuration and scores them; it never writes a
corpus directory.

**UPDATED 2026-09-25, AND THE OLD TEXT HERE IS NOW MISLEADING.** This file used
to say generation stays closed and that any improvement found here is a
reported sensitivity rather than grounds to rebuild. Generation was reopened by
author decision at the close of Stage 2h and the corpus was rebuilt as
`corpus_2026-09-25` (decision 197). The parameters below marked "currently"
refer to the values BEFORE that change: `min_q1_over_iqr` is now **0.2** and
`cv_log10_mean` is now **0.329**.

**TWO GATES NOW STAND BETWEEN A SWEEP RESULT AND A REGENERATION, and a
candidate that skips them is not a candidate.** A configuration must pass
`audits/parent_sampler_fidelity.py`, which compares the truth run's sampler
against the parent's own quantiles -- the widened configuration this stage
first tried scored WELL here and was wrong by more than 1 percent on 55 percent
of its parents -- and `audits/draft_end_to_end.py`, which runs the whole
pipeline on a draft corpus. Both were written after a regeneration passed every
sample-level check and produced a truth run with 99.98 percent errors.
Decisions 191, 192.

**AND THE OBJECTIVE ALONE IS NOT ENOUGH, for a reason this stage learned the
hard way.** Two configurations in the sweep below beat the default by improving
the objective, and both are the objective asking for the WRONG FIX: they make
the two arms agree by having the synthetic arm adopt the empirical arm's false
assumption about weights. Read the warning on `mode_coupling` before acting on
any row.

## The four parameters this stage was told to sweep

    min_q1_over_iqr      currently 0.5. Stage 2a-3 measured 0.05 and found the
                         coefficient-of-variation distance improves from 0.273
                         to 0.199 and visible-mode total variation from 0.007
                         to 0.005, with the objective flat. It did not adopt
                         it because adopting meant another regeneration for a
                         gain inside noise.
    min_mode_sd_frac     currently 0.15. A judgment with no empirical anchor:
                         the comparable empirical quantity needs a fitted
                         mixture, and 39.7 percent of empirical datasets sit
                         exactly on `gmm_em_1d`'s variance floor, so there is
                         nothing reliable to calibrate against.
    trunc_iqr_mult       currently 3.0. **The latent bug in
                         `MixtureParent.truncated_moments`, fixed in Stage
                         2a-2, would have bitten this sweep specifically**: it
                         integrated on a uniform grid and returned sd = 0
                         whenever the truncation bounds were wide relative to
                         the body, which is exactly what raising this
                         parameter produces. Confirm the fix holds across the
                         range by checking that no cell reports a degenerate
                         spread.
    mode_coupling        currently 1.0, the parameter introduced in Stage 2a
                         Part 3. At 0 a point's weight carries no information
                         about which mode it came from, which is the old
                         uncoupled behavior.

## Two more the weight model reaches

    mode_share_alpha     currently 10. The author proposes 1, which is more
                         realistic: at k = 2 the largest mode then spans 0.52
                         to 0.97 instead of 0.51 to 0.71. Stage 2f measured it
                         5.0 seed standard deviations worse on the
                         calibration, **but that comparison used the mismatched
                         weight rules and decision 141 requires it redone once
                         the arms agree.**
    point_weight_alpha   currently 1.0, the Dirichlet concentration WITHIN a
                         mode. Decision 97 established that concentration
                         alone is captured by the effective sample size while
                         COHERENCE is not, so this axis must be swept beside
                         the block structure and not instead of it; that is
                         what `audits/weight_model.py` does.

## The commands

```bash
cd audits

# the four the stage names
conda run -n compareuq python tune_configuration.py sweep '{
  "min_q1_over_iqr 0.05": {"min_q1_over_iqr": 0.05},
  "min_q1_over_iqr 0.2":  {"min_q1_over_iqr": 0.2},
  "min_q1_over_iqr 1.0":  {"min_q1_over_iqr": 1.0},
  "min_mode_sd_frac 0.05": {"min_mode_sd_frac": 0.05},
  "min_mode_sd_frac 0.10": {"min_mode_sd_frac": 0.10},
  "min_mode_sd_frac 0.25": {"min_mode_sd_frac": 0.25},
  "trunc_iqr_mult 2": {"trunc_iqr_mult": 2.0},
  "trunc_iqr_mult 5": {"trunc_iqr_mult": 5.0},
  "trunc_iqr_mult 8": {"trunc_iqr_mult": 8.0}
}'

# the two the weight model reaches
conda run -n compareuq python tune_configuration.py sweep '{
  "mode_coupling 0.0":  {"mode_coupling": 0.0},
  "mode_coupling 0.5":  {"mode_coupling": 0.5},
  "mode_share_alpha 1": {"mode_share_alpha": 1.0},
  "mode_share_alpha 3": {"mode_share_alpha": 3.0},
  "point_weight_alpha 0.3": {"point_weight_alpha": 0.3},
  "point_weight_alpha 3.0": {"point_weight_alpha": 3.0}
}'
```

## How to read the result

The objective's seed-to-seed standard deviation is **0.0066**, recorded in
decisions 47, 49, 61, 63 and 138. A configuration that moves it by less than
that has not moved it. And decision 39's failure mode is the one to watch:
a configuration can improve the statistic being watched while the corpus gets
worse on the quantity the study is built on, so read
`w_v_uw_wasserstein`'s own column and not the objective alone.
