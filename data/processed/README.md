# data/processed

## Which corpus the paper describes

**`corpus_2026-09-25`, and only that one.** `CORPUS.json` names it and the
notebooks read it from there; they never regenerate.

Every other `corpus_*` directory here is **superseded**. They are not a second
dataset, an alternative result, or a robustness arm, and no number in the paper
comes from any of them. They are kept for one reason: a corpus in this project
is immutable, so a change to generation writes a NEW directory beside the old
one and the two can be diffed file by file. That is what lets a change be
proved to have moved only what it was meant to move -- twice in this project a
corpus was REMEASURED rather than redrawn, and the evidence was that
`values.parquet`, `parents.json.gz` and `combos.csv` came back byte identical.

Two kinds sit here:

| pattern | what it is |
|---|---|
| `corpus_<date>` | a full 10,000-dataset corpus, superseded by a later one |
| `corpus_*_draft1k`, `corpus_draft_*` | a 1,000-dataset DRAFT, used to judge a candidate configuration before committing to it. **A draft must never supply a paper number** and none does |

## What is in the deposit and what is not

A corpus is about 200 MB and is **regenerable from what is tracked**, so the
large files are not committed:

    tracked        runmeta.json         the seed, the full configuration, the
                                        git commit and the library versions
                   combos.csv           the pLCA groupings
                   invalid_datasets.json  what the validity filter refused

    not tracked    values.parquet       the datasets themselves
                   metrics.parquet      their characteristics
                   parents.json.gz      how each parent was asked for
                   parents_spec.json.gz, mode_labels.parquet
                                        derived replay caches

To rebuild the active corpus from a clean clone:

    cd src && python corpus.py 2026-09-25

`data/INPUTS.sha256` pins the checksums of the large files, so a copy obtained
any other way can be proved genuine. The empirical arm needs nothing
regenerated: it is built from the frozen, tracked, checksummed EC3 extract in
`data/raw/`.
