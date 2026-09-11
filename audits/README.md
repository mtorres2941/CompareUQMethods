# audits/

One-off measurement scripts, one directory per stage. They are tracked because
the handoff quotes their numbers, and a number in a handoff that cannot be
recomputed is an assertion rather than a measurement.

They are not part of the `src/` package: nothing in `src/`, `tests/` or
`notebooks/` imports them, and they are not on the notebook `sys.path`. They
read from `data/baseline_frozen/` by default, so they keep reporting the
pre-regeneration baseline after the corpus is regenerated.

Run from inside the stage directory, with the pinned environment:

    cd audits/stage2a && python a1_collapse_corpus_damage.py

Tables land in `outputs/tables/stage2a/`.
