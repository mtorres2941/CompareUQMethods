# HANDOFF stage-2a-2 - Fresh empirical data, and the retune on it

## 1. Stage and branch

- Stage: 2a-2, fresh empirical extract and generator retune
- Branch: `stage-2a2-empirical`
- Branched from: `302099d` "Add a conciseness constraint to the standing project
  rules", on `stage-2a-generator`

| Commit | Label |
|---|---|
| `c159c51` Rebuild the empirical arm from raw values, cleaned symmetrically | **MOVES NUMBERS** |
| `ede8e2b` Retune against the new arm; weight the objective | config only |
| `1b45ca2` Regenerate as corpus_2026-09-12c; re-freeze the empirical fixture | **MOVES NUMBERS** |
| `c6b8c31` Control the minimum ADJACENT overlap, not the average; draft at 1,000 | **MOVES NUMBERS** |
| `1cba830` Score every characteristic equally; pick the range against noise | **MOVES NUMBERS** |
| `a0eb90f` Refuse spike-shaped components; show both arms in one figure | **MOVES NUMBERS** |
| this commit: count VISIBLE modes, restore Stage 2a's overlap range | **MOVES NUMBERS** |

## 2. What was asked

Take a fresh EC3 pull, validate it, freeze and checksum it as a dated immutable
input, clean it symmetrically in log space at both ends, report which of the 138
categories change, and re-run the cleaning sensitivity on the new data. Then
retune the generator configuration against the new empirical characteristics,
weighting modality and the coefficient of variation above the rest, close the
modality gap if it can be closed cheaply, regenerate once into a new dated
corpus, and report both corpora against both empirical arms.

Keep EPD-level uniform weighting as the primary analysis; the deduplicated
variant is Stage 2h's.

## 3. What was done

### 3.1 The EC3 API is gone, and what replaced it

**A fresh pull was attempted first and is not possible from this machine.** The
API returns HTTP 403:

    {"detail":"Direct API access is not allowed for private or restricted
    accounts. Please use a business account or reach out to
    support@buildingtransparency.org"}

The key is recognized. `lucidlca.ec3.check_api_token` distinguishes this case
explicitly from a bad key, and reports it as an account permission rather than a
credential problem. Nothing in this repository can fix it; Building Transparency
has to enable API access for the account.

The author chose the substitute: the consolidated EPD store at
`../EPDsFromEC3/store`, whose slice of the 138 categories was pulled on
2026-08-13 and 2026-08-14 through the LucidLCA wrapper, which paginates
correctly. It serves the purpose the fresh pull was for, on three counts. It is
five months newer than the 2026-03 data the manuscript reports. It is RAW: no
outlier rule has ever been applied to it, which is the only reason the cleaning
rule can now treat both ends of the distribution the same way. And it was pulled
with the pagination defects in `../EPDsFromEC3/PULLING_EPDS.md` already fixed.

`data/raw/ec3_raw_ecc_2026-08-14.csv.gz` is the frozen input: 2.3 MB, one row per
EPD, tracked in git and checksummed in `data/INPUTS.sha256` beside its query, its
pull dates and its source digest. This is the part that matters most for the
paper. An API pull is not reproducible by a reader even when the API works,
because EC3's contents change as declarations are issued and expire; an
archived, checksummed extract is. The empirical arm is reproducible from a clean
clone for the first time.

### 3.2 What an ECC is, and why the definition was reproduced rather than improved

The 2026-03 extraction computed `gwp / declared_unit`, converting the declared
unit with this repository's own `funcs_unit_conversion` table and keeping only
the majority unit TYPE in each category. That definition is reproduced exactly.
This stage is scoped to change the data vintage and the cleaning rule; changing
what an ECC is at the same time would make the movement in every characteristic
unattributable.

Two departures from the letter of the old code, both corrections rather than
changes of definition:

- **Each value is converted by its own unit** rather than by the category's
  majority converter. The old `consistent_units` ran every value through the
  majority type's converter, so a minority-unit record silently became NaN and
  was dropped. The set of records kept is the same either way; what changes is
  that nothing is silently NaN-ed. This is the defect `PULLING_EPDS.md` section
  4b warns about.
- **The store's pre-parsed columns are NOT used.** The store's unit table has no
  entry for `km`, so it returns NaN for the 254 of 421 `PowerCabling` records
  declared as "1 km". Parsing `declared_unit_raw` with this repository's table
  keeps them. 2.2 percent of records carry a unit this table cannot type, almost
  all of them `item`, `stück` and `unit`, which the 2026-03 extraction also
  excluded.

### 3.3 Validating the extract

A row count is not the check. `PULLING_EPDS.md` section 1 records a pull that
reached its expected row total by re-fetching page 1 fifty-four times and
reported success, so the check has to be on distinct record ids.

**Attrition, every record accounted for:**

| | records |
|---|---|
| store slice, 138 categories, latest pull each | 206,668 |
| expired at pull date, dropped | 79,694 (38.6%) |
| valid at pull date | 126,974 |
| no usable GWP or declared unit, or a minority unit type | 6,694 (5.3%) |
| **ECC values retained** | **120,280** |

The 206,668 figure reproduces Stage 2a's count of the same slice exactly.

**Duplication:** 120,280 rows, 120,280 distinct `open_xpd_uuid`, zero duplicates,
zero categories containing a duplicate, zero EPDs appearing in two of the 138
categories, and rows/unique is 1.0000 rather than an integer multiple of a page
size. The pagination is sound.

**Expired declarations.** The Aug-2026 pulls include them; the 2026-03 pull did
not, because the `ec3` library filters them silently. The author chose to match
the 2026-03 scope, valid at the pull date, so that this stage changes the vintage
and the cleaning rule rather than the population as well.

**Category-by-category, against the 2026-03 arm:** 93 grew, 11 unchanged, 34
shrank.

**Every one of the 34 shrinkages is expiry, not a short pull.** The proof
available without a live API is that the store slice counted BEFORE the validity
filter still contains at least as many records as the 2026-03 arm held. It does,
in 34 of 34.

**Definitional disagreements.** 130 of 138 category medians land within 10x of
the 2026-03 value, and the median ratio across all 138 is 1.022. Seven of the
eight that do not have more than one declared-unit type, so the two extractions
disagree about which dimension the category is measured in: `Chairs` is `item` in
the new extract and was mass in the old, because `item` has no entry in the unit
table and the old majority vote therefore fell to the kilogram-declared records.
These are reported per category in
`outputs/tables/stage2a2/TABLE_2a2_ChangeDiagnosis.csv` rather than reconciled,
because the unit type is a property of what EC3 holds today.

### 3.4 Symmetric cleaning, and the count that is no longer 138

`clean_empirical_low_end` becomes `clean_empirical_symmetric`. The log-space
3 x IQR bound now binds at both ends:

    Q1(log x) - 3 * IQR(log x)  <  log(x)  <  Q3(log x) + 3 * IQR(log x)

It removes 816 of 120,280 values, 0.678 percent: 544 at the low end and 272 at
the high end, across 60 of 138 categories. `Cement` had a value at 4.7e-10 of its
mean and `Asphalt` one at 2.6e-08; both go.

**THE EMPIRICAL ARM IS 136 CATEGORIES, NOT 138.** The manuscript states 138
throughout. `Siding` retains 1 value and `SinglePlyOther` 2, against an inclusion
threshold of 3. Both losses are expiry rather than cleaning: `Siding` holds 29
records in the store slice, of which 1 is still valid at the pull date.
Discrepancy entry 28.

### 3.5 The cleaning sensitivity, measured honestly for the first time

Stage 2a's sensitivity was measured on a store reconstruction that used
`gwp_per_kg`, against a baseline that had already been trimmed once. This one is
measured on raw values under the ECC definition the analysis actually uses.

Mean absolute metric shift against no cleaning, in standard deviations of the
uncleaned metric:

| metric | additive 3 IQR (2026-03 rule) | log 3 IQR low only (Stage 2a) | **log 3 IQR symmetric (now)** |
|---|---|---|---|
| fit_norm_SW | 0.858 | 0.205 | **0.525** |
| weight_outliers | 0.757 | 0.667 | **0.622** |
| w_v_uw_wasserstein | 0.739 | 0.586 | **0.618** |
| modality_index | 0.688 | 0.579 | **0.609** |
| entropy | 0.685 | 0.093 | **0.452** |
| fit_lognorm_SW | 0.518 | 0.596 | **0.702** |
| skewness | 0.335 | 0.155 | **0.263** |
| coeffvar | 0.273 | 0.102 | **0.227** |
| values removed | 0.85% | 0.45% | **0.68%** |

The symmetric rule is less disruptive than the additive rule it replaces on
seven of the eight characteristics, at a comparable number of values removed. It
is more disruptive than the low-end-only rule, which is expected: the low-end-only
rule leaves the right tail alone, and the right tail is where the mass is.

### 3.6 The empirical arm moved a long way, and that is the stage's main result

Every range in `src/genconfig.py` cites an empirical measurement, so a new
extract invalidates the calibration whether or not anything else changes. It did
more than that.

| characteristic | Stage 2a arm | new arm |
|---|---|---|
| categories | 138 | 136 |
| median coefficient of variation | 0.600 | **0.782** |
| log10 sd of it | 0.2913 | 0.3752 |
| maximum | 2.40 | 13.40 |
| median skewness | 1.055 | **2.060** |
| maximum skewness | 4.618 | 20.65 |
| median excess kurtosis | 1.160 | **5.758** |
| median dataset size | 37 | 53 |
| median `crit_bw_1` | 0.574 | 0.852 |
| **share Silverman calls unimodal** | **81.9%** | **49.3%** |
| BIC-selected mixture multimodal | 79.0% | 86.0% |
| fitted component overlap, median | 0.0218 | 0.0474 |
| fitted overlap, 95th pct / max | 0.4528 / 0.6719 | 0.2671 / 0.5956 |

The cause is mostly the cleaning rule rather than five months of new products.
The 2026-03 file was trimmed additively at the HIGH end before it was stored,
which cuts the right tail, and the right tail is what carries the coefficient of
variation, the skewness and the kurtosis. Restoring it moves all three.

**The multimodality row is the consequential one.** The paper's central
comparison is a KDE, which can represent a second mode, against parametric fits,
which cannot. The empirical prevalence of multimodality roughly doubled. Stage
2a's corpus was tuned to match 81.9 percent unimodal, a figure that was itself
partly an artifact of the trimming, and it is 88.4 percent unimodal as a result.

**A measurement that independently confirms the direction.** The empirical fitted
component overlap has quartiles 0.0020, 0.0474 and 0.1078, with a 95th percentile
of 0.2671. The Stage 2a configuration draws the overlap target log-uniformly on
[0.3, 1.4], which sits **entirely above the empirical 95th percentile**. Lowering
it is not only what the mode counts ask for; it is what the direct measurement of
the empirical data asks for. Both were measured on the new arm in
`audits/stage2a2/p6_empirical_envelope.py` and `p7_empirical_overlap.py`.

### 3.7 The retune, and the tuning objective

The loop now weights the characteristics instead of averaging them. Averaging
treats `n`, which the strata fix by construction, as mattering as much as
modality, which decides whether a KDE can beat a parametric fit at all.

    crit_bw_1 3.0, coeffvar 3.0, modality_index 2.0, n 0.25, all others 1.0,
    plus the mode-count total variation distance as its own term at 3.0

`modality_index` is 2 rather than 3 because it and `crit_bw_1` measure the same
property and weighting both at 3 would give modality six units of influence
rather than three. `n` is 0.25 rather than 0 so that a stratum which failed to
fill still shows up. Every run reports the objective weighted and unweighted.

What changed in the configuration, and the measurement behind each:

| field | Stage 2a | now | why |
|---|---|---|---|
| `overlap_log10_lo` | log10(0.3) | -2.5 | empirical fitted overlap 95th pct 0.2671; mode counts |
| `cv_log10_mean` | 0.011 | 0.211 | empirical median coefficient of variation 0.600 to 0.782 |
| `cv_log10_sd` | 0.2913 x 2 | 0.3752 x 2 | empirical log10 sd remeasured |
| `cv_log10_hi` | log10(3.2) | log10(16) | 3.2 no longer bracketed the empirical maximum of 13.40 |
| `EMPIRICAL_STRATUM_SHARE` | 138-dataset arm | 136-dataset arm | remeasured |

Three batches of candidates were scored at 440 datasets, the pre-flight scale
Stage 2a's handoff asks for. They are logged verbatim in
`outputs/tables/stage2a2/LOG_2a2_TuningBatch*.txt`. Batch 1 swept the overlap
lower bound alone; batch 2 swept it jointly with the coefficient of variation;
batch 3 tried four ways of recovering the lognormality that the winner costs.
**All four failed**, each making `fit_lognorm_SW` worse rather than better:

| candidate | objective | fit_lognorm_SW |
|---|---|---|
| C, chosen | 0.4261 | 1.828 |
| C + component skewness to 14 | 0.4511 | 1.797 |
| C + overlap ceiling 0.6 | 0.4377 | 1.993 |
| C + position_skew 8 | 0.4632 | 2.126 |
| C + both | 0.4772 | 2.192 |

So the cost is structural rather than a tuning artifact, and tuning stopped
there, as the prompt directs.

## 4. Numbers that moved

Every empirical characteristic moved (section 3.6) and the corpus was
regenerated several times. Nothing downstream had been computed against any of
them, which is why reopening generation was cheap.

### 4.1 The corpus, and the fact that it is a draft

**`corpus_2026-09-12i_draft1k` is active and is a DRAFT at 1,000 datasets.** The
paper needs a full 10,000 regeneration once the generator is settled. Drafts are
built with `python corpus.py <label> 1000`, 110 s against 850 s, and every
downstream check is three to five times faster. The label says `draft1k` and
`runmeta.json` carries `n_corpus`, so a draft cannot be mistaken for a paper
corpus.

### 4.2 The mistake that cost this stage, and how it was found

The retune was steered by the **Silverman** mode-count distribution, which the
new empirical arm puts at 49.3 percent unimodal against the old arm's 81.9. To
match it, the component overlap range was driven down from Stage 2a's
[0.3, 1.4] to as low as [1e-3.0, 0.5], through five configurations.

**Every one of those was a regression, and the statistics being watched said
otherwise.** Silverman's critical-bandwidth test detects structure at ANY
bandwidth, including fine structure that never appears in a plot. The empirical
datasets it calls multimodal are single right-skewed humps to look at: **94.9
percent of them have exactly one mode visible in a default-bandwidth KDE.** The
corpus was therefore rebuilt out of clearly separated humps in order to match a
count that, in real data, comes from something else.

The author reported the shapes were wrong three times, from the figures. Each
time the response was to compute another statistic of the same kind, all of
which reported the corpus as fine. What finally caught it was counting the modes
a reader can see:

| visible modes | empirical | corpus at [1e-2.5, 0.9] |
|---|---|---|
| 1 | 94.3% | 58.9% |
| 2 | 5.7% | 38.4% |
| 3 | 0% | 2.8% |

`modality.n_modes_visible` is that measure: local maxima of a Scott's-bandwidth
KDE, keeping peaks whose prominence is at least 5 percent of the tallest. It is
**the author's original `estimate_maxima` with one change**, a prominence
threshold in place of a continuous index. Stage 2a rejected that metric for
spanning only 1.000 to 1.159 across the empirical datasets, which is a real
defect in the readout; rejecting the whole idea and moving to a different
question was the error. The right arrangement is to tune against the visible
count and keep Silverman as a reported characteristic.

### 4.3 Restoring Stage 2a's overlap range, and what it recovered

Sweep on the 2026-08 arm, scoring both modality measures:

| overlap range | objective | visible TV | visible unimodal (emp 94.9%) |
|---|---|---|---|
| [1e-2.5, 0.9] | 0.4584 | 0.280 | 66.8% |
| [0.05, 1.0] | 0.4754 | 0.273 | 67.6% |
| [0.15, 1.4] | 0.4486 | 0.074 | 87.4% |
| **[0.3, 1.4], Stage 2a's** | 0.4504 | **0.022** | 92.7% |
| [0.5, 2.0] | 0.4475 | 0.029 | 97.8% |

The four objectives span 2 percent, inside the seed-to-seed noise measured in
`p10_config_noise.py`, so the objective does not choose between them and the
visible-mode distribution does. [0.3, 1.4] restored.

Full draft corpus against the 136 empirical datasets, standardized W1:

| characteristic | 12c | draft1k_f | **draft1k_i** |
|---|---|---|---|
| `fit_lognorm_SW` | 1.921 | 1.181 | **0.840** |
| `fit_norm_SW` | 0.319 | — | 0.783 |
| `crit_bw_1` | 0.524 | — | 0.715 |
| `entropy` | 0.497 | — | 0.640 |
| `skewness` | 0.642 | — | 0.617 |
| `coeffvar` | 0.314 | — | 0.387 |
| `kurtosis` | 0.299 | — | 0.303 |
| `weight_outliers` | 0.370 | — | 0.288 |
| `n` | 0.181 | — | 0.180 |
| `w_v_uw_wasserstein` | 0.550 | 0.453 | **0.131** |
| **mean W1** | 0.562 | 0.508 | **0.488** |
| visible-mode TV | — | — | **0.003** |

`w_v_uw_wasserstein` is the paper's central quantity and it is now the best-
matched characteristic in the set.

### 4.4 Three generator defects fixed along the way

1. **Components with an unbounded density.** beta with a < 1 or b < 1, and
   beta-prime with a < 1, are J-shaped: ordinary moments, infinite density at an
   endpoint, and they draw as vertical spikes. 11.3 percent of all components
   were shaped like that. `components.has_bounded_density` refuses them;
   `solve_component` returns `unbounded_density` so the target is redrawn and
   counted. Zero remain.
2. **No floor on mode width.** `min_mode_sd_frac = 0.15` requires the narrowest
   component to be at least that fraction of the parent's standard deviation.
   Before it, 29.3 percent of parents with overlap under 0.01 had a mode under
   10 percent of the dataset spread. It is a judgment, not a measurement: there
   is no reliable empirical target, because 39.7 percent of empirical datasets
   sit on `gmm_em_1d`'s reg = 1e-6 variance floor. **Owner: 2h to sweep.**
3. **`_Affine.std` was wrong by up to a factor of three**, returning `scale`
   rather than `scale * sd(shape)`. The mode-width floor was applied to the
   wrong quantity for one iteration. Caught by comparing `std()` against the
   requested sd.

### 4.5 The dataset-examples figure now compares like with like

It previously drew the synthetic PARENT DENSITY, exact and sharp, against a
reader's memory of real data, which is only ever seen smoothed. It now shows a
KDE of the values for both arms, the parent density as a thin line where one
exists, and a row of empirical datasets underneath.

Measured, the synthetic data was never the spikier arm: peak-to-median KDE
height has an empirical median of 5.9, a 95th percentile of 222 and a maximum of
5e39, against a synthetic median of 3.0 and a maximum of 33.

### 4.6 Two hypotheses tested and rejected, recorded so they are not retried

- **Making the synthetic truncation multiplicative** to match the empirical
  rule. It is incompatible with the additive shift that controls the
  coefficient of variation: as the shift grows, q3/q1 tends to 1 and the bounds
  collapse onto the interquartile range. Truncated mass rose from a median of
  0.149 to 0.247 and the achieved coefficient of variation went to 0.000.
  Reverted; `trunc_rule` keeps both and its docstring holds the numbers.
- **Empirical multimodality as a duplicate-value artifact.** EC3 categories
  contain many identical declarations, so tied values were a plausible cause.
  They are not: collapsing exact ties moves the multimodal share from 52.3 to
  51.5 percent, and datasets with under 5 percent ties are as multimodal (47.9)
  as those with over 20 percent (52.9).

## 5. Open questions and flags

### Carried forward

| Item | Owner | Status |
|---|---|---|
| Take a fresh EC3 pull and clean symmetrically | 2a-2 | **RESOLVED, with a caveat.** Cleaned symmetrically from raw values. The pull is a frozen 2026-08 store slice, not a live API pull, because the API is now closed to this account |
| Bandwidth rule, KL1/KL2 inconsistency | 2h | STILL OPEN |
| `logfit_offset` | 2b, swept in 2h | STILL OPEN |
| Dependent sampling | 2e | STILL OPEN |
| Overlap area alongside W1 | 2c | STILL OPEN |
| Shapiro-Wilk vs Shapiro-Francia | 2f | STILL OPEN |
| `(1-capecc)` divisor | 2g | STILL OPEN |
| Scoring grid includes zero | 2c or 2e | STILL OPEN |
| W1 has no complexity penalty | 2c | STILL OPEN |
| `weighted_quantile` must stay fixed before Silverman in 2h | 2h | STILL OPEN |
| Entry 13, support (0, inf), needs author confirmation | - | **STILL OPEN.** Stage 2a built on it and so does this stage |
| Corpus marginally under-multimodal, 11.6 vs 18.1 | 2a-2 | **SUPERSEDED.** The target itself moved; see section 3.6 |
| The `(1 - capecc)` divisor, figure sizes, git history | 2g, 3, 4 | STILL OPEN, untouched |
| Deduplicated empirical variant | 2h | STILL OPEN. Primary stays EPD-level uniform, per the author |
| `mode_share_alpha` at 10 | 2h | STILL OPEN |
| `trunc_iqr_mult` sweep | 2h | STILL OPEN |
| Kurtosis undefined in stratum 1 | 2f | STILL OPEN |
| Notebooks 2 and 3 never run against the active corpus | 2b | STILL OPEN, deliberately |

### Needing the author's decision, in priority order

1. **Regenerate at 10,000 once the author is satisfied with the generator.**
   The active corpus is a 1,000-dataset draft. Nothing downstream should be run
   against it.
2. **The tuning objective now scores both modality measures.** `n_modes_visible`
   is the one to steer by; `n_modes_silverman` stays as a reported
   characteristic. Any future sweep that optimises the Silverman distribution
   alone will repeat this stage's mistake.
3. **`min_mode_sd_frac = 0.15` is a judgment with no empirical anchor**, for the
   reason in 4.4. **Owner: 2h.**
4. **The acceptance criterion as written was not met by `corpus_2026-09-12c`**
   and the four-way table in `TABLE_2a2_FourWayComparison.csv` is from that
   corpus, so it is stale. It should be re-run against the final corpus.

### New in Stage 2a-2

- **EC3 direct API access is closed to this account.** Restoring it is an author
  action: support@buildingtransparency.org, business account. Until then the
  empirical arm cannot be refreshed, only re-derived from the local store.
  Discrepancy entry 30.
- **The empirical arm is 136 categories.** Entry 28.
- **Some EC3 categories are not one product population.** With the right tail no
  longer cut, `PowerCabling` spans 1.7e-05 to 242 times its own mean over 400
  values, `Aggregates` 3.2e-06 to 265, `Insulation` 3.0e-04 to 99. Their
  coefficients of variation are 13.4, 10.3 and 7.8 against a median of 0.78.
  These are not cleaning failures: the log-space IQR of such a category is
  genuinely enormous, so a 3 x IQR bound is permissive on it. They dominate the
  upper tail of every characteristic and therefore stretch the envelope the
  corpus is asked to cover. **No stage owns this.** Entry 31.
- **`Elevators` keeps a value at 1.2e-07 of its mean** after cleaning, for the
  same reason. It is the clearest single case of the point above.
- **The empirical characteristics were an artifact of the cleaning rule to a
  degree nobody had measured.** Entry 29. This is the argument for having done
  the extract at all, and it is worth stating in the paper rather than only in a
  handoff.
- **Matching modality costs lognormality, structurally.** `fit_lognorm_SW` 0.865
  to 1.921. Four attempts to recover it made it worse. Real ECC datasets are 49
  percent multimodal while keeping a median Shapiro-lognormal statistic of
  0.937; the generator reaches the same mode count by separating components,
  which is a different shape. **Owner: 2h**, and it is the most interesting open
  question the stage produced: what generative structure gives a gentle second
  mode on a lognormal body?
- **5.6 percent of the corpus has six or more modes against an empirical 0.7
  percent.** Same cause. **Owner: 2h.**
- **The Silverman unimodal share carries a few points of estimator noise**: the
  same 136 datasets read 49.3 percent at 100 bootstrap replicates and 45.6 at
  60. Matching modality to within 2 points is at the resolution of the
  measurement. Any later stage quoting a modality share must quote `nboot` with
  it.
- **`CompareUQMethods_SUPP_DatasetExamplesByStratum.png` is misleading.** Its
  x-axis is set by the parent's truncation bounds rather than by the data, so
  datasets that are perfectly reasonable read as needles beside an empty tail. I
  misread it that way myself before measuring, and a reviewer will too. The
  measurement is in section 4.5. **Owner: 3.**
- **`audits/stage2a/a6_empirical_source.py` still refers to `mode_count_est`**,
  renamed in Stage 2a, so it would fail if re-run. Harmless, one-off audit
  script, noted so it is not rediscovered as a defect.

## 6. Inputs and outputs

**Read:** `reports/` in full, `CLAUDE.md`, `CONTEXT.md`, `src/`, `notebooks/`,
`../EPDsFromEC3/PULLING_EPDS.md` in full, `../EPDsFromEC3/store/README.md`,
`../EPDsFromEC3/ec3_fulldownload.ipynb` for the 2026-03 query, and
`../LucidLCA/lucidlca/ec3.py`.

**Written:** `data/raw/ec3_raw_ecc_2026-08-14.csv.gz` and its runmeta;
`audits/stage2a2/` (p1 to p7); `outputs/tables/stage2a2/`; this file.

**Modified:** `src/empirical.py`, `src/datageneration.py`, `src/coverage.py`,
`src/genconfig.py`, `audits/stage2a/b5_tune_configuration.py`,
`notebooks/01_CompareUQ_CreateData.ipynb`, `CONTEXT.md`, `README.md`,
`CLAUDE.md`, `data/INPUTS.sha256`, `reports/MANUSCRIPT_discrepancies.md`
(entries 28 to 31), `tests/`.

**Not touched:** `data/processed/dct_realeccs_trimmed.json`, kept as the record
of the data the manuscript reports; `corpus_2026-09-12b`, kept as the
comparison; notebooks 2 and 3; the manuscript.

## 7. Next stage

Stage 2b, unchanged in scope, with one addition: its first task is still to run
notebooks 2 and 3 against the active corpus, which have never been run against
any corpus later than the pre-regeneration one.

**Generation is closed again.** It was reopened once, by decision, because
nothing downstream had been computed against `corpus_2026-09-12b`. That stops
being true the moment notebook 2 runs.

**Read section 5's numbered list before running notebook 2.** The first item
may change the corpus, and it is far cheaper to settle it now than after 2b has
produced results against this one.

A fresh EC3 pull is being taken in the `EPDsFromEC3` repository. When it lands,
the empirical arm is rebuilt by adapting `audits/stage2a2/p1_build_raw_extract.py`
to the new file, validated with `p2` and `p3`, and the tuning loop re-run.
Whether that justifies a third regeneration is an author decision. The author's
expectation, recorded 2026-09-12, is that a month of new EPDs will not change
much.

**Do not query the EC3 API while that pull is running.** EC3 rate limits per
account rather than per process, and a concurrent request is what truncated a
ready-mix pull to 9 percent of the category while reporting success.

## 8. State at the end of the session

| Item | State |
|---|---|
| Branch | `stage-2a2-empirical`, 3 commits |
| Tests | 126 passing |
| Active corpus | `corpus_2026-09-12c`, seed 42, 10,000 + 50 probe, 0 failed, 0 rejected |
| Previous corpus | `corpus_2026-09-12b`, kept on disk as the comparison |
| Empirical arm | 136 categories, `data/raw/ec3_raw_ecc_2026-08-14.csv.gz`, tracked and checksummed |
| Notebook 1 | runs clean end to end against the active corpus |
| Notebooks 2 and 3 | NOT run. Still Stage 2b's first task |
| Fixtures | `TABLE_EmpiricalECCMetrics.xlsx` re-frozen at 136 rows, `SHA256SUMS.txt` updated and verifying. The two W1 tables are still pinned to the PRE-regeneration corpus and are Stage 2b's to re-freeze |
| `data/INPUTS.sha256` | rows added for the raw extract and for `corpus_2026-09-12c` |
| Open decisions | 3, listed in section 5, the first of which may change the corpus |
