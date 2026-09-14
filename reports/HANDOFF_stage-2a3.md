# HANDOFF stage-2a-3 - Split the heterogeneous categories, and check the envelope

## 0. STATUS, read this first

**THE SPLIT WAS APPLIED AND THEN WITHDRAWN. THE ARM IS 136 UNSPLIT CATEGORIES.**
`empirical.SPLIT` is False. Sections 3.1 to 3.8 below describe the split as it
was built and are kept because the measurements in them are the evidence for
whatever replaces it; **section 3.1a is the part a later stage should act on.**

The axis was the DECLARED UNIT, and the author rejected it on review, 2026-09-14:
"Separating by declared unit doesn't quite seem reasonable. Why is it strange
that some aggregates might be declared per 1 kg and some per 1000 kg? That still
might be the same material." That is correct, and this stage's own evidence
already said so: where the split appeared to work it was because the declaration
convention happened to CORRELATE with contamination, and the one category where
the correlation failed, `ConcreteAdmixtures`, was flagged as "the weakest of the
six" and kept anyway because the rule was uniform. A uniform rule on the wrong
axis is still the wrong axis.

**What survives and is worth keeping:** the constraint (metadata only, never the
values), the axis survey in 3.1 and 3.1a, the screen, `q1`'s verification of the
extract, the weight rekeying of 3.5, the corpus, and the coverage finding in 4.3,
which is the most consequential thing this stage found and is independent of the
split.

**What is still open:** which axis, if any. Section 3.1a gives the candidates
with numbers and a recommendation.

## 1. Stage and branch

- Stage: 2a-3, split the EC3 categories that are not one product population
- Branch: `stage-2a3-categories`
- Branched from: `25d0937` "Remove an unfounded claim that EC3 API access was
  closed", on `stage-2a2-empirical`

| Commit | Label |
|---|---|
| `41eb7ae` Split the EC3 categories that are not one product population | **MOVES NUMBERS** |
| `0011913` Retune `cv_log10_sd`, regenerate as `corpus_2026-09-14b` | **MOVES NUMBERS** |
| `e72c286` Make the input manifest verify again | records only |
| this commit: band the declared unit by RATIO, and show the split in notebook 1 | **MOVES NUMBERS** |

## 2. What was asked

Some EC3 categories are not a single product population. Substantiate and apply
the splits, on record metadata only and never on the ECC values. Choose what to
split by a stated screen applied to all 136 categories, not case by case.
Measure what the split does to the empirical envelope, and reopen generation
only if a stated noise criterion says so. Correct two records: the claim that
EC3 API access is closed, and the open question of folding in a newer pull.

This is the last pre-2b stage. Nothing after it reopens generation or the
empirical extract.

## 3. What was done

### 3.1 The three candidate split axes, and which one exists

The prompt named the declared unit type first and the EC3 category path second.
Neither is available as a split axis on this arm, and establishing that is a
finding about EC3 rather than a limitation of the analysis.

`audits/stage2a3/q1_rebuild_slice.py` reconstructs the 2026-08 store slice,
verifies it against the frozen extract, and writes the per-record metadata the
frozen extract does not carry. Verification, which the script refuses to write
without: 206,668 records in the slice, 120,280 in the majority-unit subset
against a frozen 120,280, **identical record ids and identical categories**, ECC
agreeing to a worst relative difference of 8.9e-13. The frozen file is CSV text,
so the comparison is at round-trip precision rather than bitwise.

| axis | verdict |
|---|---|
| **A. EC3 category path** (`category_key`) | **NOT AVAILABLE.** It equals the queried category for all 123,060 usable records in all 138 categories, and the finer `category` field is empty throughout. There is no subcategory |
| **B. Declared unit TYPE** | **ALREADY APPLIED.** The extraction restricts each category to the unit type most of its products use, so every dataset in the arm holds one unit type by construction. 106 of 138 categories contain records of another type, but those 2,780 records were dropped when the extract was built. Reinstating them would change what an ECC is and would add about 126 mostly tiny datasets; it would not divide any existing population |
| **C. Declared unit SCALE** | **THE AXIS THAT WORKS.** The declared quantity converted to the unit type's canonical unit, expressed as a RATIO to the way most of the category declares itself, and banded in groups of three decades, which is one SI prefix step. See section 3.3a for why the ratio is not optional |
| D. A product-type field | **NOT AVAILABLE.** All 106 store columns were searched for the screened categories. Nothing is populated for 90 percent or more of records with more than one level except declarer attributes: program operator, PCR, jurisdiction, plant specificity, uncertainty factor. A declarer is not a product population |

### 3.1a WHAT THE METADATA CAN ACTUALLY SUPPORT, and the recommendation

Measured after the declared-unit axis was withdrawn, on the 2026-08 store slice.
This is the section a later stage should act on.

**The EC3 category tree is the finding.** `../EPDsFromEC3/store/category_tree.csv`
holds 104 nodes with parents, children and a leaf flag, and it was not consulted
when the split was built. **19 of the 136 categories are NON-LEAF nodes**: their
records are EPDs that EC3 placed at the parent rather than in any child, which
makes them residual mixtures BY CONSTRUCTION rather than product populations. For
15 of the 19, the children are already separate datasets in the arm.

| parent | n | CV | children in the arm |
|---|---|---|---|
| `Insulation` | 666 | **7.65** | `BlanketInsulation`, `BlownInsulation`, `BoardInsulation`, `FoamedInPlace` |
| `Masonry` | 35 | 1.86 | `Brick` |
| `CeilingPanel` | 240 | 1.71 | `AcousticalCeilings` |
| `Steel` | 576 | 1.57 | 4 of 7 |
| `StructuralSteel` | 105 | 1.35 | `HollowSections` |
| `Aluminium` | 181 | 1.14 | `AluminiumExtrusions` |
| `Cladding` | 135 | 1.07 | 4 of 4 |
| `MembraneRoofing` | 168 | 0.82 | 3 of 3 |
| `Finishes`, `Flooring`, `ManufacturingInputs`, `CementitiousMaterials`, `ColdFormedSteel`, `ThermalMoistureProtection`, `Concrete` | 4 to 213 | 0.29 to 0.76 | yes |
| `PaintingAndCoating`, `FireAndSmokeProtection`, `Openings`, `PrecastConcrete` | 12 to 546 | 0.41 to 1.36 | **none queried** |

`Insulation` is the worst dataset in the arm and it is a parent whose four
children are all already present. That is the author's own example, and the tree
answers it without any text matching.

**Structured product properties.** One field qualifies, and it is the author's
other example. `concrete_compressive_strength_28d` is populated on 90 to 96
percent of records in eight categories (`ReadyMix` 86,995 records at 89.9
percent, `CementGrout`, `FlowableFill`, `Shotcrete`, `CMU`, `ConcretePaving`,
`OilPatch`, `Concrete`), in psi, clustering on the standard classes 3000, 3500,
4000, 4500, 5000 and 6000. The only other broadly populated field is
`density_value`, at 80 percent or more in 21 categories, and it is a poor axis:
density is a consequence of the material rather than a specification choice, and
where the declared unit is volume it partly determines the ECC, so splitting on
it edges toward splitting on the values.

**Free text.** `name` is always present and `description` on 87 percent. Tested
on `Insulation` with a keyword list for mineral wool, EPS, XPS, PIR/PUR,
cellulose, wood fibre, phenolic and aerogel: **350 of 666 records match exactly
one type, 311 match none and 5 match more than one.** Splitting on that would
leave 47 percent of the category unassigned.

#### The recommendation

1. **Drop the 15 parent-node residual datasets whose children are in the arm.**
   Metadata only, no judgment, no text, and it removes the worst offender. It is
   not a split: a parent node is not a product category, and its products are
   already represented by its children. The arm becomes **121 datasets**.
2. **Do NOT split the concrete categories by strength**, despite the field being
   there. `ReadyMix`'s coefficient of variation is **0.29** against an arm median
   of 0.82: after each dataset is normalized to its own mean it is one of the
   TIGHTEST categories in the arm, because ECC scales with strength and
   normalization removes most of that. Splitting the eight categories by strength
   class would produce 47 datasets from 8, making concrete about a quarter of the
   whole arm, and would split categories that show no evidence of needing it.
   That is the over-zealousness the author warned against, quantified.
3. **Leave the remaining high-dispersion categories whole and state the
   limitation.** After the parent drop, `Aggregates`, `PowerCabling`, `Grouting`,
   `Chairs`, `ConcreteAdmixtures` and `Elevators` remain above a coefficient of
   variation of 3. None is a parent node and none has a populated product-type
   field; their heterogeneity is EC3 misclassification visible only in free-text
   names. The alternative to leaving them is to EXCLUDE them, which is an
   inclusion criterion rather than a split, but it reads the values to decide and
   it would bias the arm toward low dispersion, on the exact dimension the study
   measures. That trade is the author's to make.

**Nothing in 1 to 3 is implemented.** It needs the author's decision, and it
moves every empirical number when it lands.

### 3.2 The screen

Stated before the split was applied, and applied to all 138 extracted
categories rather than to the three already named:

> A category is selected if its unweighted coefficient of variation, after the
> arm's cleaning rule, exceeds **3.0**. The arm's median is 0.77 and its 95th
> percentile 2.72, so 3.0 selects the extreme upper tail. The unweighted form is
> used deliberately: it does not depend on the Dirichlet weight draw, so the
> screen is reproducible from the frozen extract alone.

**The coefficient of variation is a SCREEN, never a boundary.** It decides which
categories are examined. Where a split falls is decided entirely by metadata.
That distinction is what keeps a paper about modality and dispersion from
arguing in a circle, and it is the first thing a reviewer will look for.

Two other clauses were evaluated on all 138 and are reported rather than acted
on: more than one declared unit type present selects 106 categories and splits
none, for the reason in row B above; more than one EC3 subcategory selects zero.

**The screen selects 7 of 136. Six split, into twelve populations. One could
not be.**

### 3.3 The splits, with their substantiation

Every split is on `declared_unit_raw`, the declared unit recorded on the EPD.
The full table with one substantiating sentence per population is
`outputs/tables/TABLE_EmpiricalCategorySplit.csv`, written by notebook 1.

| category | CV | populations | n |
|---|---|---|---|
| `Aggregates` | 13.6 | `Aggregates [1000 kg]` / `Aggregates [1 kg]` | 350 / 34 |
| `PowerCabling` | 12.8 | `PowerCabling [1 km]` / `PowerCabling [1 m]` | 251 / 148 |
| `Grouting` | 4.3 | `Grouting [1 kg]` / `Grouting [1000 kg]` | 211 / 14 |
| `Chairs` | 3.9 | `Chairs [1000 kg]` / `Chairs [1 kg]` | 55 / 33 |
| `ConcreteAdmixtures` | 3.6 | `ConcreteAdmixtures [1 kg]` / `[1000 kg]` | 92 / 26 |
| `Elevators` | 3.2 | `Elevators [1 t]` / `Elevators [1 kg]` | 17 / 3 |
| `Insulation` | 7.6 | **NOT SPLIT** | 666 |

Four records fall in bands too small to form a dataset and are dropped, which is
the rule the arm already applies to a category with fewer than three values: one
`Aggregates` at 0.007 kg, two `Elevators` at 41,765 kg, one `PowerCabling` at
0.02 m. They are listed in the split table with `DROPPED` in the name.

### 3.3a The band must be a RATIO, and the first version was not

Found by the author on review, by reading the table this stage produced, which
is the fourth time in this project that looking at the output caught what the
audits did not.

The first version banded `log10(du_value)` directly, on the canonical unit. **The
canonical unit for length is the INCH.** 0.65 m is 25.6 in and 1 m is 39.4 in,
which straddle a decade boundary, so `PowerCabling` came out as THREE
populations with a spurious three-record `[0.65 m]` split off from `[1 m]` -- a
factor of 1.5 apart. The rule's own justification says a split separates
functional units at least three orders of magnitude apart, and its output
contradicted that.

The band is now computed on `du_value / du_reference`, where the reference is the
modal declared quantity in the category. Band 0 always holds the way most of the
category declares itself and the other bands are powers of a thousand away from
it, so the justification holds by construction and no arbitrary absolute scale
enters. `PowerCabling` becomes two populations, 251 per kilometre and 148 per
metre; the one record at 0.02 m falls three bands out and is dropped.

**Effect: the arm is 142 datasets, not 143.** Every other category is unchanged
in structure. The corpus was NOT regenerated for it: `corpus_2026-09-14b` scored
against the corrected arm gives 0.2305 against 0.2256 for the arm before the
correction, a movement of 0.0049 or 0.74 seed-to-seed standard deviations, and
`cv_log10_sd` moves from 0.3536 to 0.3519, which is 0.0017. Both are inside
noise, so a third regeneration would buy nothing.

**The paper-facing sentence, uniform across the six:** *within the category, EPD
declarations state the functional unit at scales differing by at least three
orders of magnitude, and a declaration per tonne and one per kilogram are
different functional units, so the groups are treated as separate populations
rather than pooled.*

**The product evidence behind it, from the record names**, which is
substantiation and not the split rule. It is worth putting in the paper because
it says what the scale band is a proxy for:

- `Aggregates [1 kg]` holds adhesives, screeds, porcelain stoneware and resin;
  `Aggregates [1000 kg]` holds aggregate. The category is contaminated.
- `Grouting [1000 kg]` holds precast sandwich panels, prestressing steel strand
  and lightweight concrete panels; `Grouting [1 kg]` holds plasters, renders and
  skimcoats.
- `Chairs [1000 kg]` holds asphalt, hollowcore slabs and column elements;
  `Chairs [1 kg]` holds chairs and furniture.
- `PowerCabling [1 km]` holds North American AWG and kcmil power cable;
  `[1 m]` holds European mm2 and kV building cable.
- `Elevators [1 kg]` holds three OTIS elevators declared per kilogram, giving
  20,812 kgCO2e/kg. These are declaration errors, and the split isolates them
  rather than correcting them.
- `ConcreteAdmixtures` is the weakest of the six: both bands hold admixtures,
  and the split separates declaration conventions rather than products.

**`Insulation` could not be split, and that is the honest outcome the prompt
allows.** Its 666 area-declared records are all near 1 m2, carry one EC3
category path and one declared-unit type, and its heterogeneity is product
thickness and R-value. `thickness_value` is populated for 0.7 percent of the
arm's records. It is left whole with a coefficient of variation of 7.6, and the
reason is recorded in the split table itself.

### 3.4 The count that replaces 136

**THE EMPIRICAL ARM IS 142 DATASETS DRAWN FROM 136 EC3 CATEGORIES.** Both
numbers belong in the manuscript: 142 is what every per-dataset statement
counts, 136 is how many EC3 categories they were drawn from. Discrepancy entry
28 is updated; it had just replaced 138 with 136.

### 3.5 The weight draw was coupled to the iteration order, and it is not any more

Splitting six categories moved the weighted metrics of 130 datasets that were
not touched. The cause: `empirical.prepare` drew each dataset's Dirichlet
weights from one Generator in sorted order, so inserting `Aggregates [1 kg]`
shifted the draw for every dataset after it alphabetically.

This is the same argument the function's own docstring already made one level
up, about not letting a figure earlier in the notebook change the weights. It
had not been applied to the dataset list itself. Weights are now keyed by
dataset NAME: `_dataset_rng` folds the name through SHA-256 with base entropy
drawn once from the passed Generator, so the arm still moves with the notebook
seed, nothing touches global numpy state, and a dataset's weights are a property
of that dataset.

**The two movements, separated.** After rekeying, splitting moves the 130 shared
datasets by EXACTLY ZERO on every column. Section 4 gives the size of the
rekeying movement, which is large and is a finding in its own right.

### 3.6 The envelope, before and against after

`audits/stage2a3/q2_envelope_before_after.py`, and
`audits/stage2a2/p6_empirical_envelope.py` and `p7_empirical_overlap.py` re-run
on the split arm. Both columns below use name-keyed weights, so the only
difference between them is the split.

| quantity | unsplit (136) | split (142) |
|---|---|---|
| coefficient of variation, median | 0.7570 | 0.7675 |
| log10 sd of it | 0.3800 | **0.3536** |
| minimum / maximum | 0.0062 / **14.34** | 0.0062 / **11.20** |
| skewness, median | 1.7317 | 1.7395 |
| skewness, minimum / maximum | -2.2049 / **28.36** | -2.2049 / **14.03** |
| excess kurtosis, median / maximum | 5.49 / **835.3** | 5.31 / **201.4** |
| dataset size, median / maximum | 53 / 86,770 | 47 / 86,770 |
| `crit_bw_1`, median / maximum | 0.7946 / 4.176 | 0.7898 / 3.277 |
| Silverman unimodal, nboot = 100 | 49.26% | 50.35% |
| **visible modes: 1 / 2 / 3+** | **94.85 / 5.15 / 0%** | **95.10 / 4.90 / 0%** |
| entropy, median | 3.0112 | 2.9041 |
| `weight_outliers`, median | 0.0440 | 0.0447 |
| `fit_norm_SW` / `fit_lognorm_SW`, median | 0.8186 / 0.9394 | 0.8140 / 0.9384 |
| `w_v_uw_wasserstein`, median | 0.0988 | 0.1011 |
| stratum shares s1 / s2 / s3 / s4 | .0956 / .5588 / .2941 / .0441 | .1049 / .5664 / .2797 / .0420 |
| BIC-multimodal share (`p7`) | 86.0% | 88.1% |
| fitted overlap, median / 95th | 0.0474 / 0.2671 | 0.0446 / 0.2897 |

The split column above was measured before the band was corrected to a ratio
(section 3.3a), so it describes the 143-dataset arm. Remeasured on the final
142-dataset arm the differences are small and in the same direction: maximum
coefficient of variation 8.96 rather than 11.20, maximum skewness 11.97 rather
than 14.03, maximum excess kurtosis 195.1 rather than 201.4, median dataset size
48.5, Silverman unimodal 50.7 percent, visible unimodal 95.07 percent, stratum
shares .0986 / .5704 / .2817 / .0423. `TABLE_2a3_EnvelopeBeforeAfter.csv` holds
the final numbers; the BIC and overlap rows were not remeasured, since the
correction merges a 3-record population into a 146-record one.

**The split cuts the upper tail and leaves the body alone.** Every maximum falls
substantially, because the widest categories were the heterogeneous ones; every
median moves by less than 0.11; the visible-mode distribution moves by 0.0025,
a twelfth of the seed-to-seed noise. Silverman's share is quoted with nboot as
required, and its 1.1-point movement is at the resolution of the estimator.

### 3.7 Regeneration: the criterion said yes, and the retune found one field

`audits/stage2a3/q3_corpus_vs_split_arm.py`. `corpus_2026-09-13b` scored against
both arms with the tuning objective, every characteristic weighted equally,
against the seed noise from `p10_config_noise.py` (objective sd 0.0066, mode
total variation 0.029):

| | unsplit (136) | split (142) | moved |
|---|---|---|---|
| weighted objective | 0.2239 | 0.2352 | 0.0113 = **1.72 sd** |
| Silverman mode TV | 0.2106 | 0.2001 | 0.0105, inside noise |
| visible mode TV | 0.0044 | 0.0019 | 0.0025, inside noise |

**The criterion fails on the objective, so the tuning loop was re-run.** Exactly
one genconfig field cites a measurement that moved: `cv_log10_sd`, which cites
the arm's log10 standard deviation of the coefficient of variation, 0.3752 to
0.3536. Nothing else moved: the coefficient-of-variation range still brackets
the arm with margin at [0.004, 16] against [0.0062, 11.20]; `cv_log10_mean` is a
deliberate population-versus-sample offset and the arm's own log10 mean moved by
0.0025; the overlap range is set by the visible-mode distribution, which moved
by 0.0025. `EMPIRICAL_STRATUM_SHARE` also moved and was updated, but it is a
post-stratification weight and never a generation parameter.

**Honest accounting of what the retune is worth.** At the 440-dataset pre-flight
scale the change improves the objective from 0.2307 to 0.2274, a movement of
0.0033 against a noise standard deviation of 0.0066. That is HALF the noise. The
measurement is adopted because it is the measurement the parameter cites, not
because the improvement is distinguishable from a different seed.

**With it, the criterion passes.** The 1,000-dataset draft
`corpus_2026-09-14a_draft1k` scores 0.2183 against the unsplit arm and 0.2219
against the split arm: a movement of 0.0036, **0.55 sd**, inside noise on all
three measures. The split no longer displaces the calibration.

### 3.8 The regeneration

`corpus_2026-09-14b` is the corpus Stage 2b should use, and
`data/processed/CORPUS.json` points at it. 10,000 datasets plus a 50-dataset
probe set, seed 42, 0 failed parents, 0 rejected by the validity filter, 862 s.
Notebook 1 was re-run against it, which is what writes `combos.csv` into the
corpus directory; `corpus.py` does not.

**Four-way, both corpora against both arms**, as Stage 2a-2 did:

Scored on the FINAL 142-dataset arm, after the band correction of section 3.3a:

| corpus | arm | objective | mean W1 | Silverman TV | visible TV |
|---|---|---|---|---|---|
| `2026-09-13b` | unsplit (136) | 0.2239 | 0.2471 | 0.2106 | 0.0044 |
| `2026-09-13b` | **split (142)** | 0.2403 | 0.2684 | 0.1965 | 0.0022 |
| `2026-09-14b` | unsplit (136) | 0.2147 | 0.2385 | 0.1909 | 0.0010 |
| **`2026-09-14b`** | **split (142)** | **0.2305** | **0.2586** | **0.1768** | **0.0032** |

Against the split arm, which is the arm the analysis uses, the objective
improves by 0.0098, or 1.48 noise standard deviations, and mean W1 across the
ten characteristics from 0.2684 to 0.2586. `TABLE_2a3_FourWayComparison.csv` and
`TABLE_2a3_PerCharacteristic_<label>.csv` hold the detail.

**A note on reading `q3`'s own verdict line.** It prints "RETUNE and regenerate
once" for `corpus_2026-09-14b` as well. That is not a second call to
regenerate. The quantity it tests is how far the objective moves when the
REFERENCE changes, and part of that gap is an irreducible difference between
two reference sets rather than a mismatch any corpus can close. It is the right
input to the decision exactly once, for the corpus that predates the split. The
script now says so in its own output.

## 4. Numbers that moved

### 4.1 Rekeying the Dirichlet weights, and it is larger than the split

Redrawing the weights of the same 136 datasets from the same distribution,
changing nothing else, on the 130 datasets present before and after:

| characteristic | max absolute movement |
|---|---|
| excess kurtosis | 365.8 |
| skewness | 9.88 |
| coefficient of variation | 4.09 |
| **`w_v_uw_wasserstein`** | **1.02** |
| mean (weighted) | 1.17 |
| `crit_bw_1` | 0.563 |
| entropy | 0.553 |
| `fit_norm_SW` | 0.391 |
| `fit_lognorm_SW` | 0.371 |
| `weight_outliers` | 0.339 |
| `modality_index` | 0.195 |

**Every UNWEIGHTED column is bit-identical across the two realizations**, which
is the proof that no value changed and only the weights did.

This is a real finding and it is written up as discrepancy entry 32. A single
Dirichlet realization moves the paper's central per-dataset quantity by up to
1.02 in absolute terms. The arm-level DISTRIBUTION is far more stable, and that
is what the study rests on, but the manuscript does not currently make the
distinction. **Owner: 2h**, which already owns "multiple weight realizations".

### 4.2 The split

After rekeying, zero on the 130 shared datasets. One consequence that is not
zero: cleaning now runs per POPULATION rather than per category, so each
population's interquartile range is computed on its own values. 823 of 120,277
values are removed against 816 of 120,280 before, 545 low and 278 high against
544 and 272. Twelve populations replace
six categories; three records are dropped for falling in bands below the
three-value threshold. Arm 136 to 142 datasets. Envelope movement in section
3.6.

### 4.3 The corpus

`corpus_2026-09-14b` replaces `corpus_2026-09-13b` as the active corpus.
Section 3.8 has the four-way table. Mean standardized W1 across the ten
characteristics, both scored against the final split arm: 0.2684 to 0.2586.

**Coverage, and it is the most important number in this stage.** The manuscript
claims the synthetic corpus covers the empirical characteristic space and
extends beyond it on every side. Counting empirical datasets that fall outside
the synthetic range, over the ten characteristics:

| | unsplit arm (136) | split arm (142) |
|---|---|---|
| `corpus_2026-09-13b` | **10** | 7 |
| `corpus_2026-09-14b` | 13 | **9** |

Net for this stage, 10 to 9. The split improves coverage, because the
categories it separates were the extreme ones; the retune costs a little,
because narrowing `cv_log10_sd` narrows the synthetic range.

**But coverage was already broken before this stage, and nobody had measured
it.** Decision 29 records 100 percent coverage on all nine characteristics, and
that was measured on the Stage 2a arm, whose maximum coefficient of variation
was 2.40. Stage 2a-2 rebuilt the arm from raw values and the maximum became
13.40, while the synthetic maximum is 2.58. The claim has been false since Stage
2a-2 and this stage is the first to check it. See section 5.

### 4.4 Configuration

| field | before | after | the measurement it cites |
|---|---|---|---|
| `cv_log10_sd` | 0.3752 x 2 | 0.3536 x 2 | arm log10 sd of the coefficient of variation |
| `EMPIRICAL_STRATUM_SHARE` | 13/76/40/6 of 136 | 15/81/40/6 of 142 | share of the arm in each size stratum; post-stratification only |

### 4.5 The input manifest verifies again

`shasum -a 256 -c data/INPUTS.sha256`, the command the manifest documents, had
two rows that failed by design and now has none. `data/processed/CORPUS.json` is
a POINTER that changes with the active corpus and the manifest's own note
already said it should not have been pinned; the superseded runmeta row is the
one this stage corrected. Both are COMMENTED OUT rather than deleted, so the
values stay on the record. No baseline row was touched.

### 4.6 Fixtures

`tests/fixtures/TABLE_EmpiricalECCMetrics.xlsx` re-frozen, 136 to 142 rows, with
`SHA256SUMS.txt` updated in the same commit. Both changes above move it.
`TABLE_EmpiricalECCMetricsAndW1.xlsx` is still the 136-row fixture and is
**stale by design**: notebook 2 has never been run against any recent corpus,
and re-freezing it is Stage 2b's first task. 128 tests pass.

## 5. Open questions and flags

### Carried forward

| Item | Owner | Status |
|---|---|---|
| Bandwidth rule, KL1/KL2 inconsistency | 2h | STILL OPEN |
| `logfit_offset` | 2b, swept in 2h | STILL OPEN |
| Dependent sampling | 2e | STILL OPEN |
| Overlap area alongside W1 | 2c | STILL OPEN |
| Shapiro-Wilk vs Shapiro-Francia | 2f | STILL OPEN |
| `(1-capecc)` divisor | 2g | STILL OPEN |
| Scoring grid includes zero | 2c or 2e | STILL OPEN |
| W1 has no complexity penalty | 2c | STILL OPEN |
| `weighted_quantile` must stay fixed before Silverman in 2h | 2h | STILL OPEN |
| Entry 13, support (0, inf), needs author confirmation | - | **STILL OPEN.** Four stages have now built on it |
| Deduplicated empirical variant | 2h | STILL OPEN. Primary stays EPD-level uniform |
| `mode_share_alpha` at 10 | 2h | STILL OPEN |
| `trunc_iqr_mult` sweep | 2h | STILL OPEN |
| Kurtosis undefined in stratum 1 | 2f | STILL OPEN |
| `min_mode_sd_frac = 0.15` has no empirical anchor | 2h | STILL OPEN |
| Six or more modes, 5.6 pct of corpus vs 0.7 empirical | 2h | STILL OPEN |
| `SUPP_DatasetExamplesByStratum.png` x-axis is misleading | 3 | STILL OPEN |
| Figure sizes, git history | 3, 4 | STILL OPEN, untouched |
| `audits/stage2a/a6_empirical_source.py` refers to `mode_count_est` | - | STILL OPEN, harmless |
| Notebooks 2 and 3 never run against the active corpus | 2b | STILL OPEN, deliberately. **Now the oldest item in the project** |
| Some EC3 categories are not one product population | 2a-3 | **RESOLVED.** Section 3.3. Six split, `Insulation` left whole with the reason recorded |
| Fold in a newer EC3 pull | 2a-2 | **RESOLVED.** No. Decision 44, the arm is frozen |
| EC3 API is closed to this account | 2a-2 | **RESOLVED.** It is not. Decision 45 |

### New in Stage 2a-3

- **A single Dirichlet weight realization moves per-dataset weighted metrics a
  long way.** Section 4.1, discrepancy entry 32. **Owner: 2h.** This is the one
  item here that could change a headline number.
- **EC3 carries no subcategory for these records.** Discrepancy entry 33.
  `category_key` equals the queried category for all 123,060 usable records and
  the finer `category` field is empty throughout. It is worth one sentence in
  the paper, because a reader will ask why product type was not used.
- **`ConcreteAdmixtures` is the weakest of the six splits.** Both bands hold
  admixtures, so it separates declaration conventions rather than products. It
  is kept because the screen and the rule are applied uniformly and picking it
  out afterwards would be exactly the case-by-case judgment the stage was told
  to avoid. If the author wants it pooled, that is a one-line change and it
  should be recorded as a deliberate exception.
- **`PowerCabling [0.65 m]` was a defect, not a dataset.** Section 3.3a. It is
  gone. Recorded because it is the kind of thing that only shows up when
  somebody reads the table.
- **`Elevators [1 kg]` isolates three declaration errors** rather than a product
  population: 20,812 kgCO2e/kg for an elevator. The split confines them to a
  three-value dataset instead of letting them set the whole category's spread.
  Whether an obvious declaration error should be dropped rather than isolated is
  an author decision, and no stage owns it.
- **The retune was worth half the noise.** Section 3.7. Recorded so a later
  stage does not read `cv_log10_sd = 0.3536 * 2` as a measured improvement.
- **THE COVERAGE CLAIM IS FALSE AND HAS BEEN SINCE STAGE 2a-2.** Decision 29
  records 100 percent coverage of the empirical characteristic space on all
  nine characteristics, approved by the author from
  `CompareUQMethods_FIG_MetricCoverage.png`. It was measured on the Stage 2a
  arm, whose maximum coefficient of variation was 2.40. Stage 2a-2 rebuilt the
  arm from raw values, the maximum became 13.40, and nothing re-checked
  coverage. The synthetic maximum is 2.18, so **six empirical datasets have a
  coefficient of variation the corpus never reaches**: `PowerCabling [1 m]`
  at 11.20, `Insulation` at 6.05, `ConcreteAdmixtures [1 kg]` at 3.56,
  `Grouting [1 kg]` at 3.24, `DampproofingAndWaterproofing` at 2.50 and
  `WallFinishes` at 2.20. Two more are uncovered on `fit_norm_SW` and one on
  `n` (`ReadyMix`, 86,770 values against a corpus ceiling of 9,999, which is
  decision 19 and is covered by the probe set instead).

  **The cause is not the draw range**, which reaches 16. It is that the
  coefficient of variation is a POPULATION target and the characteristic
  measured is the SAMPLE value, which runs systematically low on a right-skewed
  distribution; `genconfig.cv_log10_mean` already carries an offset for this.
  Stage 2a-2 recorded that only 41.7 percent of coefficient-of-variation targets
  are met. Closing the gap means either raising the offset further or fixing the
  solve, and both are generation changes.

  **This stage did not act on it, deliberately.** Generation is closed, it is
  not this stage's scope, and acting on it would have meant a second
  regeneration on a question nobody has decided. **It needs an author decision,
  and it is the one item here that could require reopening generation again.**
  The honest alternative to reopening is to state the limitation: the corpus
  covers the empirical characteristic space with margin except at the top of the
  coefficient of variation, where six of 142 datasets sit beyond it, four of
  them categories that are not one product population. Figure
  `CompareUQMethods_FIG_MetricCoverage.png` and decision 29 must both be
  revisited either way. **Owner: unassigned. Raise it before Stage 2b runs
  notebook 2.**

## 6. Inputs and outputs

**Read:** `CLAUDE.md`, `CONTEXT.md`, `reports/` in full, `src/`,
`notebooks/01`, `../EPDsFromEC3/store/epd_index.csv.gz`.

**Written:** `src/categorysplit.py`; `audits/stage2a3/` (README, q1 to q3);
`data/raw/ec3_record_metadata_2026-08-14.csv.gz`;
`outputs/tables/TABLE_EmpiricalCategorySplit.csv`;
`outputs/tables/stage2a3/`; `data/processed/corpus_2026-09-14a_draft1k/` and
`corpus_2026-09-14b/`; this file.

**Modified:** `src/empirical.py`, `src/genconfig.py`,
`notebooks/01_CompareUQ_CreateData.ipynb`, `CLAUDE.md` (decisions 43 to 45, the
roadmap), `CONTEXT.md`, `data/INPUTS.sha256`,
`data/raw/ec3_raw_ecc_2026-08-14_runmeta.json`,
`reports/MANUSCRIPT_discrepancies.md` (entries 28, 31 updated; 32 and 33 new),
`tests/fixtures/TABLE_EmpiricalECCMetrics.xlsx` and `SHA256SUMS.txt`,
`outputs/tables/TABLE_EmpiricalECCMetrics.xlsx`, `.gitignore`,
`data/processed/CORPUS.json`.

**Not touched:** notebooks 2 and 3; the manuscript; `dct_realeccs_trimmed.json`;
the superseded corpora.

## 7. Next stage

**Stage 2b, the lognormal.** Its first task is unchanged and is now the oldest
outstanding item in the project by a wide margin: **run notebooks 2 and 3
against `corpus_2026-09-14b`**, and re-freeze
`tests/fixtures/TABLE_EmpiricalECCMetricsAndW1.xlsx` and
`TABLE_SyntheticECCMetricsAndW1.xlsx` in the same commit that moves them.

**BOTH INPUTS ARE NOW CLOSED.** Generation was reopened here by the stated
criterion, used once, and is closed. The empirical arm is frozen at the 2026-08
extract by decision 44. Nothing after this stage moves either one.

### Read this before touching the generator or the arm again

The four habits in `reports/HANDOFF_stage-2a2.md` section 7 all still hold, and
this stage adds a fifth.

5. **A change to the LIST of datasets is a change to every dataset, unless the
   randomness is keyed by identity.** Splitting six categories moved the
   weighted metrics of 130 untouched ones, because the weights were drawn in
   iteration order. This is the same failure as cleaning the two arms by
   different rules: an incidental difference gets attributed to the change under
   study. Anything that is drawn per dataset should be keyed by the dataset.
