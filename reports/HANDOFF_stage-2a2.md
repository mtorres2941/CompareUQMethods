# HANDOFF stage-2a-2 - Fresh empirical data, and the retune on it

## 1. Stage and branch

- Stage: 2a-2, fresh empirical extract and generator retune
- Branch: `stage-2a2-empirical`
- Branched from: `302099d` "Add a conciseness constraint to the standing project
  rules", on `stage-2a-generator`

| Commit | Label |
|---|---|
| `c159c51` Rebuild the empirical arm from raw values, cleaned symmetrically | **MOVES NUMBERS** |
| (filled in below as the stage proceeds) | |

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

## 4. Numbers that moved

Section 3.6 is the summary; every empirical characteristic moved. Nothing
downstream of the empirical arm had been computed against
`corpus_2026-09-12b`, which is why reopening generation was cheap here and will
not be again.

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
