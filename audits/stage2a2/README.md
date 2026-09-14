# Stage 2a-2 audits

One-off measurement scripts. Each prints its result and writes a table to
`outputs/tables/stage2a2/`. Run from this directory under the `compareuq`
environment.

| script | answers |
|---|---|
| `p1_build_raw_extract.py` | Builds the frozen raw EC3 extract in `data/raw/`. Refuses to overwrite a dated file. Adapt it when a fresh pull becomes available |
| `p2_validate_extract.py` | Is the extract sound? Attrition, distinct-id duplication, per-category comparison against the 2026-03 arm |
| `p3_diagnose_changes.py` | Why did a category grow, shrink or move? Separates expiry from a short pull, and unit-type disagreements from real change |
| `p4_cleaning_report.py` | What the cleaning rule removes, and how much the choice of rule moves each characteristic |
| `p5_four_way.py` | Both corpora against both empirical arms, so a change can be attributed to the data or to the generator |
| `p6_empirical_envelope.py` | The empirical measurements every range in `src/genconfig.py` cites |
| `p7_empirical_overlap.py` | Empirical component overlap from a BIC-selected mixture |
| `p8_spikiness.py` | Is a dataset a needle in a long support? Both arms, same estimator |
| `p9_mode_realism.py` | Mode widths and adjacent-pair gaps. Reports how many empirical mode widths are pinned at the EM regularizer floor, which is 39.7 percent |
| `p10_config_noise.py` | Is the gap between two candidate configurations real, or seed noise? Run this before believing any sweep result |

## The one that matters most

`p10_config_noise.py`. The tuning loop scores a candidate on 440 generated
datasets, and the within-configuration standard deviation of the objective is
0.0066 while typical differences between candidates are the same size. Several
configuration choices in this stage were made on a single draw and were reading
noise.

## What these scripts could not see

Three real defects were found by the author looking at a figure, after these
audits reported the corpus as fine:

- components with an UNBOUNDED density (J-shaped beta and beta-prime). Their
  moments are ordinary, so neither value concentration (`p8`) nor component
  standard deviation (`p9`) can detect them;
- modes far narrower than the dataset they sit in;
- a visible-mode distribution badly mismatched while the Silverman distribution
  matched.

`modality.n_modes_visible` exists because of the third. The general lesson is in
`reports/HANDOFF_stage-2a2.md` section 7.
