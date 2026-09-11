# Regression fixtures

Reference artifacts for the Stage 1 refactor. They exist so that a phase
labelled NEUTRAL can be proven not to have changed any result. They are not a
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

## What is deliberately absent

There is no pLCA fixture here. NB3 writes no table at all, and its Monte Carlo
draws are unseeded, so at this point its results are neither persisted nor
reproducible. Phase 1 persists them as an archive; a genuine pLCA regression
fixture becomes possible only after Phase 2 establishes seeding. See
`tests/fixtures/plca/README.md` once that exists.
