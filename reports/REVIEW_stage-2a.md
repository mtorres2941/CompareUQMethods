# How to review Stage 2a

Stage 2a rewrote the synthetic data generator and regenerated the corpus, so
every number downstream has moved. This is a route through it in the order that
lets you stop early and still have checked the things that matter most.

Budget roughly 90 minutes for all five passes. Passes 1 and 2 are the ones to
do if you only do two.

---

## Read this first

The generator is tuned by comparing the DISTRIBUTION of every statistical
characteristic between the two arms, not by checking that the synthetic range
contains the empirical values. Mean Wasserstein-1 distance across the ten
characteristics is 0.354, in empirical standard deviations. Mode counts:
81.9 / 14.5 / 2.9 percent for one, two and three modes empirically, against
88.4 / 9.7 / 1.5 synthetic.

Three things to look at first, in this order:

1. `outputs/figures/CompareUQMethods_SUPP_DatasetExamplesByStratum.png` - ten
   datasets per size stratum, each against the distribution it came from.
2. `outputs/tables/TABLE_DistributionComparison.csv` - every characteristic,
   worst first.
3. `outputs/tables/TABLE_ModalityComparison.csv` - mode counts, both arms.

`audits/stage2a/b5_tune_configuration.py` is the tuning loop. Re-run it after
any change to generation.

## Pass 1: the claims, before any code (20 min)

Read `reports/HANDOFF_stage-2a.md`. It is written to be read alone.

Read it with these four questions in hand:

1. **Section 4, "Numbers that moved."** Do you accept that the empirical
   weighting effect more than doubling (0.0594 to 0.1329) is a correction and
   not a new choice? This is the single largest change to a headline number.
2. **Section 4.3.** Two regenerations were discarded. Do you agree they should
   have been?
3. **Section 4.4.** Eight defects were found while building. Four of them
   (`ppf` collapse, zero-mass components, overlap quantization, the CV probe)
   would have silently corrupted the corpus. Does the account convince you they
   are actually fixed?
4. **Section 5, "Open questions and flags."** Everything I could not close is
   there. The three that most need your judgment are marked.

**If something in here is wrong, stop and say so.** Everything below is
downstream of it.

---

## Pass 2: does the corpus look like your data? (20 min)

This is the question the whole stage exists to answer, and it is the one you
are best placed to judge, because you know what an ECC dataset looks like.

Open **`notebooks/01_CompareUQ_CreateData.ipynb`** and read from the markdown
heading **"Do the synthetic datasets look like the real ones?"** to the end.
You do not need to run it; the outputs are the tables below.

Then look at two things on disk:

- **`outputs/figures/CompareUQMethods_FIG_MetricCoverage.png`** - the 138
  empirical datasets plotted inside the synthetic cloud, six metric pairs.
  Orange points are real materials, blue is synthetic, red circles are
  empirical datasets the synthetic corpus does not cover. There are two, both
  named in `outputs/tables/TABLE_CoverageFigureStats.csv`.
- **`outputs/tables/TABLE_MetricCoverage.csv`** - the same thing as numbers.
  `empirical_covered` is 1.000 on all nine statistical metrics.

**The honest weak spot to look at hardest:** the synthetic median coefficient
of variation is 0.487 against an empirical 0.593, the largest remaining
disagreement after modality. The `Q3 + 3*IQR` cleaning rule, which the
empirical data also obey, caps how much right tail can survive. Stage 2h should
sweep `trunc_iqr_mult`.

Also worth your eye: skewness and kurtosis reach much further in the synthetic
corpus than the empirical one. That is small-sample noise at n = 3 to 9 rather
than the generator inventing exotic shapes, but you should decide whether that
much margin is what you want.

---

## Pass 3: is the generator defensible to a reviewer? (25 min)

Read in this order. Each file has a docstring at the top saying what it does
and why, in prose, before any code.

1. **`src/genconfig.py`** - every generation parameter in one dataclass. Each
   field's docstring says which empirical measurement its range comes from.
   **This is the file to argue with.** If a range looks wrong, it is wrong here
   and nowhere else.
2. **`outputs/tables/TABLE_1_GenerationParameters.csv`** - Table 1 for the
   paper. One row per parameter: what was configured, which empirical number it
   derives from, what the corpus achieved. The last column is read back from
   the per-dataset record, so a setting that was asked for and missed shows up.
3. **`src/generator.py`**, function `draw_parent` - the 40 lines that build one
   dataset's parent. The comment block in the middle explains the two wrong
   ways the mixture was placed before this one.
4. **`src/components.py`** - the docstring explains why four distribution
   families are needed rather than three.
5. **`src/mixture.py`**, function `_pair_overlap` - only if you want to see the
   overlap calculation.

**What to push on:** `overlap_log10_lo/hi`, `cv_log10_mean`, `cv_log10_sd` and
`position_skew` were all set by fitting to the empirical distributions with
`audits/stage2a/b5_tune_configuration.py` rather than derived from anything.
That is the intended process, but they are the most arguable numbers in the
stage and the tuning loop is the thing to re-run if any of them look wrong.

---

## Pass 4: do the tests actually test anything? (15 min)

```bash
conda activate compareuq
python -m pytest tests/ -q          # 126 tests, about 75 seconds
```

Then skim the test NAMES, which are written as claims:

```bash
python -m pytest tests/ --collect-only -q | sed 's/.*:://' | sort
```

The five that carry the most weight:

- `test_parent_cdf_matches_the_pipeline_it_describes` - the parent CDF against
  400,000 draws through the real generator. **This is what Stage 2c depends on.**
- `test_market_weighted_parent_is_a_real_population_object` - Part 3's fix.
- `test_zero_coupling_collapses_the_two_parents` - states the circularity
  problem as an identity.
- `test_moment_target_is_hit_exactly` - components have the skewness and
  kurtosis asked for.
- `test_validity_filter_does_not_reject_for_being_unusual` - the replacement
  for the 27.5 percent filter.

---

## Pass 5: reproduce something yourself (10 min)

Pick any one:

```bash
# What produced the corpus in use: seed, every setting, commit, library versions
cat data/processed/corpus_2026-09-11c/runmeta.json

# The pre-regeneration baseline is intact and provable
cd data/baseline_frozen && shasum -a 256 -c INPUTS.sha256   # expect 8 OK

# The seeding-collapse measurement, recomputed from the frozen old data
cd audits/stage2a && python a2_collapse_effective_corpus.py

# Generate 50 datasets yourself and look at them (about 5 seconds)
python -c "
import sys; sys.path.insert(0,'src')
import numpy as np, genconfig, generator
rng = np.random.default_rng(1)
for i in range(3):
    x, w, rec = generator.generate_dataset(genconfig.DEFAULT, 200, rng)
    print(f'k={rec[\"k\"]} overlap={rec[\"overlap_achieved\"]:.4f} '
          f'cv_target={rec[\"cv_target\"]:.3f} cv_achieved={rec[\"cv_achieved\"]:.3f} '
          f'mean={x.mean():.6f}')
"
```

---

## What is NOT done, so you do not go looking for it

- **Notebooks 2 and 3 have not been run against the new corpus.** They are
  rewired and they parse, but their fixtures still point at the old data. This
  is Stage 2b's first task. Until it runs, `outputs/tables/TABLE_PLCAResults.csv`
  and 20 of the 21 remaining figures are from the pre-regeneration corpus.
- **No fitting method changed.** The lognormal offset, the KDE bandwidth and
  the W1 scoring target are all untouched, by design: they belong to 2b, 2c and
  2h.
- **The manuscript was not edited.** Nine new discrepancies are logged in
  `reports/MANUSCRIPT_discrepancies.md` as entries 19 to 27.

---

## The fastest possible review

If you have ten minutes:

1. `reports/HANDOFF_stage-2a.md` section 4, "Numbers that moved."
2. `outputs/figures/CompareUQMethods_FIG_MetricCoverage.png`.
3. `outputs/tables/TABLE_1_GenerationParameters.csv`.

Those three tell you whether the corpus is right and whether the reasons are
written down.
