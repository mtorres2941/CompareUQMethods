# audits/

One-off measurement scripts. They are tracked because the decision log quotes
their numbers, and a number in the log that cannot be recomputed is an
assertion rather than a measurement. Each script is named for what it measures,
and the decision that rests on it names the script.

They are not part of the `src/` package: nothing in `src/`, `tests/` or
`notebooks/` imports them, and they are not on the notebook `sys.path`. Several
read from `data/baseline_frozen/` by default, so they keep reporting the
pre-regeneration baseline after the corpus is regenerated.

They are NOT part of the analysis. The notebooks reproduce every number in the
paper on their own, and nothing in `outputs/` at the top level is written by
anything here. An audit writes only under `outputs/tables/audits/`.

Run from the repository root, with the pinned environment:

    conda run -n compareuq python audits/bandwidth_rules.py

Tables land in `outputs/tables/audits/`.
