# audits/stage2a3

One-off measurement scripts for Stage 2a-3: splitting the EC3 categories that
are not one product population, and checking whether the split moved the
empirical envelope far enough to invalidate the generator calibration.

Run from the repository root with the pinned environment.

| script | answers |
|---|---|
| `q1_rebuild_slice.py` | reconstructs the 2026-08 store slice and REFUSES to write unless it reproduces the frozen ECC extract record for record; writes the metadata sidecar that lets the split axes be audited |
| `q2_envelope_before_after.py` | every quantity `src/genconfig.py` cites, unsplit arm against split arm |
| `q3_corpus_vs_split_arm.py` | the active corpus scored against both arms with the tuning objective, against the seed-to-seed noise from `../stage2a2/p10_config_noise.py`. This is the criterion for reopening generation |

`q3` caches the corpus's own mode counts, which take about 25 minutes for 10,000
datasets and never change once a corpus is written. Delete
`outputs/tables/stage2a3/CACHE_modality_<label>.npz` to force a recount.

`../stage2a2/p6_empirical_envelope.py` and `p7_empirical_overlap.py` read the
arm through `empirical.prepare`, so they measure the SPLIT arm without
modification. Both were re-run in this stage.
