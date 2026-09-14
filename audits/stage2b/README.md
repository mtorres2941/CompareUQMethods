# audits/stage2b

One-off measurement scripts for Stage 2b: the physical-plausibility ceiling on
the empirical arm, and the lognormal fitting work.

Run from the repository root with the pinned environment.

| script | answers |
|---|---|
| `r1_plausibility.py` | the ceiling's gate: how many records and cleaned values it removes, which datasets lose them, what it does to their characteristics, the ten highest and lowest records per declared-unit type for author review, and how many records the extraction excludes for a non-positive GWP |
| `r2_calibration_after_ceiling.py` | the active corpus scored against the arm before and after the ceiling, with the Stage 2a-3 objective, against the 0.0066 seed-to-seed gate |
| `r3_weight_realization_noise.py` | how much of `r2`'s movement is a redrawn Dirichlet vector rather than a change in the data. Six datasets change length under the ceiling and therefore get a fresh draw; entry 32 |

`r2` and `r3` import `corpus_modality` and `objective` from
`../stage2a3/q3_corpus_vs_split_arm.py` rather than restating them, so the
criterion is the same object it was in Stage 2a-3 and cannot drift.
