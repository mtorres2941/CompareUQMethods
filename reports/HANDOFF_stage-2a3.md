# HANDOFF stage-2a-3 - Split the heterogeneous categories, and check the envelope

## 0. STATUS, read this first

**THE CATEGORIES ARE RESOLVED INTO SPECIFIABLE PRODUCTS. THE ARM IS 149
DATASETS. THE ACTIVE CORPUS IS `corpus_2026-09-14d`.**

This stage tried three things and the third is the one that stands. Sections 3.1
to 3.8 are the record of the first two and are kept because their measurements
are what ruled them out; **section 3.1c is what was actually implemented.**

| attempt | axis | outcome |
|---|---|---|
| 1 | declared unit, screened on the coefficient of variation | **REJECTED by the author.** A declaration per kilogram against one per tonne is a declaration convention, not a different product |
| 2 | drop the EC3 residual bins only | **INCOMPLETE.** Right for insulation and steel, but it left concrete pooled and I argued against splitting it from the wrong statistic |
| 3 | three rules: drop residual bins, split concrete by specified strength, split insulation by material type | **IMPLEMENTED** |

**The criterion this stage was given was dispersion, and that was wrong.** The
author's correction, 2026-09-14: "This exercise isn't about fixing dispersion at
all, it's separating material categories meaningfully." Splitting `ReadyMix` by
compressive strength moves its coefficient of variation only from 0.29 to 0.27
and is still obviously right, because 4000 psi and 5000 psi concrete are
different products and strength is the primary characteristic a structural
engineer specifies concrete by. A dataset here stands for one material choice in
a probabilistic LCA, so the test is whether a category is something a specifier
could name, not whether it is tight. **Do not reintroduce a dispersion screen.**

**The constraint that did survive all three attempts:** a split may read only
metadata carried on the EPD record or on EC3's category tree, never the ECC
values. That is what keeps a study of modality and dispersion from arguing in a
circle.

**Also found here and independent of any of it:** the coverage claim is false and
has been since Stage 2a-2. Section 4.3, discrepancy entry 34. It is the most
consequential thing in this stage and it is still unowned.

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

#### 3.1b Which axis actually makes a category homogeneous

`audits/stage2a3/q4_split_axis_evidence.py`,
`outputs/tables/stage2a3/TABLE_2a3_SplitAxisEvidence.csv`. The test is not
whether a field exists but whether splitting on it reduces the WITHIN-dataset
coefficient of variation. A split that leaves the dispersion where it was has
not separated anything, whatever the field is called. Two author hypotheses were
tested; one holds and one does not, and the refutation is the more useful.

**Concrete by 28-day compressive strength: the field is real, the effect is
small.** `ReadyMix`, 86,995 records, strength stated on 89.9 percent:

| class | records | CV |
|---|---|---|
| whole category | 86,995 | **0.29** |
| <3000 psi | 3,968 | 0.30 |
| 3000-3999 | 20,797 | 0.25 |
| 4000-4999 | 31,124 | 0.24 |
| 5000-5999 | 14,400 | 0.24 |
| >=6000 psi | 7,906 | 0.25 |
| unstated | 8,800 | 0.47 |
| **record-weighted within-class** | | **0.27** |

0.29 to 0.27. Strength class explains almost none of `ReadyMix`'s spread,
because each dataset is normalized to its own mean and the ECC of concrete
scales with strength, so normalization removes most of the between-class
difference before it is measured. The other seven concrete categories behave the
same way: `Shotcrete` 0.21 to 0.19, `ConcretePaving` 0.26 to 0.23, `CMU` 0.34 to
0.33, `CementGrout` 0.37 to 0.32, `FlowableFill` 0.83 to 0.84, which is worse.

The case for splitting concrete by strength is therefore NOT that the category
is heterogeneous. It is that a specifier picks a strength class, so a
strength-class dataset is a more faithful unit of analysis for a pLCA. That is a
real argument and it is the author's; it is just not a dispersion argument, and
the study's characteristic distributions will barely move if it is applied.

**Insulation by material type: REFUTED.** The author's hypothesis was EPS, XPS,
cellulose and so on. Matched against name and description:

| type | records | CV |
|---|---|---|
| whole category | 666 | **7.65** |
| **MineralWool** | **342** | **6.78** |
| XPS | 36 | 0.87 |
| PIR/PUR | 21 | 1.69 |
| EPS | 14 | 1.13 |
| unclassified | 228 | 1.80 |

Mineral wool alone, more than half the category, still has a coefficient of
variation of 6.78. Material type does not separate this category.

**Insulation by declared THICKNESS: that is the axis.** The category is declared
per square metre, and emissions per square metre scale with thickness:

| band | records | CV |
|---|---|---|
| whole category | 666 | **7.65** |
| <40 mm | 33 | 1.10 |
| 40-79 mm | 67 | 1.56 |
| 80-119 mm | 109 | 0.98 |
| 120-199 mm | 107 | 0.78 |
| >=200 mm | 72 | 0.60 |
| **no thickness in the name** | **278** | **6.91** |

Every band with a stated thickness is between 0.60 and 1.56. All of the
dispersion is in the 278 records whose thickness cannot be parsed. The extremes
say the same thing directly: the smallest value is "1 m2 silicate coating
(110 g/m2)" at 0.0003 of the category mean and the largest is a stone wool
mattress at 99 times it. **A coating and a 320 mm board are both "insulation per
square metre" and are not the same quantity.**

Two problems with using it. The thickness comes from a REGULAR EXPRESSION over
the product name, not a field: the store's `thickness_value` is populated for
ZERO of these 666 records. And it parses for only 388 of 666, leaving a
278-record residual at 6.91, which is no better than where we started.

**What EC3's own children already achieve.** The four child categories are
separate datasets in the arm and are all well behaved: `BoardInsulation` 335
records CV 1.19, `BlanketInsulation` 319 records CV 1.18, `BlownInsulation` 24
records CV 1.05, `FoamedInPlace` 16 records CV 1.68. 694 records of insulation,
already clean, already in the arm. The 666-record parent is the bin EC3 did not
classify, and it is the only badly behaved insulation dataset.

#### 3.1c THE RULES AS IMPLEMENTED

Settled with the author on 2026-09-14. `src/categorysplit.py`; the table is
`outputs/tables/TABLE_EmpiricalCategorySplit.csv`, written by notebook 1, and
notebook 1 cells 10 and 11 display it.

**Rule 1, drop the residual bins.** A category that is a NON-LEAF node of EC3's
category tree holds the EPDs EC3 did not place in any of its children, so it is a
residual bin by construction rather than a product. Where those children are
themselves categories in this arm, the material is already represented and the
bin is dropped. **15 dropped, 2,473 records.** `Insulation` (666, CV 7.65),
`Steel` (576, 1.57), `CeilingPanel` (240, 1.71), `Concrete` (213),
`Aluminium` (181), `MembraneRoofing` (168), `Cladding` (135),
`StructuralSteel` (105), `ColdFormedSteel` (67), `Masonry` (35),
`CementitiousMaterials` (24), `Flooring` (17), `Finishes` (15),
`ManufacturingInputs` (13), `ThermalMoistureProtection` (4).

This is what answers insulation and steel at once, with no new machinery. The
steel children already in the arm are exactly the distinctions the author named:
`HotRolled`, `ColdFormedSteel`, `PlateSteel`, `Hollow`, `RebarSteel`,
`DeckingSteel`, `WireMeshSteel`, `SteelSuspensionAssembly`, `Coil`. EAF against
BOF is NOT available: `steel_making_route_eaf` is populated on 0.01 percent of
records.

**Four parents are KEPT** because no child of theirs is in the arm, so dropping
them would remove the material from the study: `PrecastConcrete` (546),
`FireAndSmokeProtection` (23), `PaintingAndCoating` (19), `Openings` (12). Their
heterogeneity is a stated limitation.

**Rule 2, concrete by specified 28-day compressive strength.** `ReadyMix`,
`Shotcrete`, `ConcretePaving`, `CMU`, split at 3000, 4000, 5000 and 6000 psi,
which cut between the standard classes rather than through them. Records with no
stated strength become an explicit `[strength not stated]` dataset. **23 datasets
from 4**, one population dropped for holding 2 records. `CementGrout`,
`FlowableFill` and `OilPatch` carry the field just as well and are NOT split:
they are not specified by strength in building design, and the author's
instruction was to leave the rarely specified categories alone. `CONCRETE` in
`categorysplit.py` is the one place to change that.

**Rule 3, insulation by material type.** `BoardInsulation`, `BlanketInsulation`,
`BlownInsulation`, `FoamedInPlace`, matched against the product name and
description with the fixed pattern list in `MATERIAL_TYPES`. **13 datasets from
4**, 9 populations dropped below three records. The largest groups are
`BoardInsulation [type not stated]` 131, `BlanketInsulation [mineral wool]` 185,
`BlanketInsulation [type not stated]` 129, `BoardInsulation [mineral wool]` 78.

**"Type not stated" is kept as a dataset, not dropped.** It is 131 of 335
`BoardInsulation` records and discarding it would lose 40 percent of the
category. The paper has to say what it is: board insulation EPDs whose name and
description do not state a material.

**Thickness was tested and rejected**, which is the "as we are able" clause
biting. It is the axis that explains the `Insulation` BIN's dispersion, every
parsed band landing between 0.60 and 1.56 against the bin's 7.65, but that bin is
dropped by rule 1, and on the remaining categories a thickness parses for only 96
of 335 board and 150 of 319 blanket records; crossed with type it leaves 8 viable
groups covering 73 of 335. The store's own `thickness_value` is populated for
ZERO of these records.

#### The recommendation that preceded them

Superseded by the measurements in 3.1b. Revised, and each item now carries the
number that decides it:

1. **`ReadyMix` by strength class, if the author wants the unit of analysis to
   match a specification choice.** 6 datasets from 1, arm 136 to 141. It is
   cheap, it is a structured field at 89.9 percent, and it is confined to the
   highest-volume material rather than applied to all eight concrete categories,
   which would make concrete a quarter of the arm. Be clear in the text about
   what it buys: 0.29 to 0.27, not a homogeneity fix.
2. **`Insulation`: the parent is EC3's unclassified bin, and the clean insulation
   data is already in the arm.** Three options, with what each costs:
   (a) drop the parent, keeping the four children, 694 records at CV 1.05 to
   1.19, arm 136 to 135 and 666 records discarded;
   (b) split by thickness parsed from the product name, giving five bands at CV
   0.60 to 1.56 plus a 278-record residual still at 6.91, arm 136 to 141;
   (c) leave it, and state that one dataset in the arm pools a 110 g/m2 coating
   with a 320 mm board.
   **Recommended: (a).** It needs no text parsing, discards only the bin EC3
   itself could not classify, and insulation stays represented by four datasets.
3. **The other 14 parent-node residual datasets**: same argument, same choice,
   and the author has said `PowerCabling`, `Grouting`, `Chairs` and
   `ConcreteAdmixtures` are not worth touching because they are rarely specified.
   `Steel` (576 records, CV 1.57), `CeilingPanel` (240, 1.71), `Cladding` (135,
   1.07) and `Masonry` (35, 1.86) are the ones that matter by volume.

**Nothing is implemented.** It needs the author's decision and it moves every
empirical number when it lands.

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

### 4.3 The corpus, and an honest account of the third regeneration

**`corpus_2026-09-14d` is the corpus Stage 2b should use**, and
`data/processed/CORPUS.json` points at it. Seed 42, 886 s, **9,999 datasets plus
a 50-dataset probe set, not 10,000**: one parent failed to solve under the new
coefficient-of-variation parameters and was reported rather than approximated,
which is decision 22 working as intended. Every earlier corpus had
`n_failed_parent` 0.

Three corpora were generated in this stage, which is two more than the standing
rule allows, and the reason is that the empirical arm changed twice after the
first: `corpus_2026-09-14b` for the withdrawn declared-unit split, and
`corpus_2026-09-14d` for the rules that stand. Notebooks 2 and 3 had still never
run, so nothing downstream was invalidated by any of them.

| corpus | arm | objective | mean W1 | Silverman TV | visible TV |
|---|---|---|---|---|---|
| `2026-09-14b` | unsplit (136) | 0.2147 | 0.2385 | 0.1909 | 0.0010 |
| `2026-09-14b` | resolved (149) | **0.2037** | 0.2296 | 0.1367 | 0.0115 |
| `2026-09-14c_draft1k` | resolved (149) | 0.2013 | 0.2270 | 0.1406 | 0.0056 |
| **`2026-09-14d`** | **resolved (149)** | **0.2075** | 0.2326 | 0.1513 | 0.0128 |

**The retune made the match marginally WORSE, by 0.0038 or 0.58 seed-to-seed
standard deviations, and it is kept anyway.** That needs saying plainly rather
than buried. `corpus_2026-09-14b` scores better against the resolved arm, but its
`cv_log10_sd` of 0.3536 was measured on the declared-unit split arm that was
withdrawn and no longer exists; the resolved arm reads 0.3919. A parameter whose
cited measurement is of a discarded arm cannot be defended in the paper. The
difference is inside noise, the parameters now cite the arm actually in use, and
that is the whole justification.

**I misread the sign of this comparison once mid-stage** and told the author the
corpus scored 0.2257 against the resolved arm when it scored 0.2037. The
criterion in `q3` reports an absolute movement; the split arm was scoring BETTER,
not worse, throughout. Corrected here and in the four-way table.

### 4.3a Coverage, the most important finding in this stage

The manuscript claims the synthetic corpus covers the empirical characteristic
space and extends beyond it on every side. It does not, and it has not since
Stage 2a-2. On the 149-dataset arm against `corpus_2026-09-14d`, counting
empirical datasets outside the synthetic range over the ten characteristics:
**11 uncovered dataset-metric pairs of 1,490.**

| characteristic | uncovered | which |
|---|---|---|
| `coeffvar` | 5 | `Aggregates`, `Chairs`, `Elevators`, `Grouting`, `PowerCabling` |
| `fit_norm_SW` | 3 | `Aggregates`, `Grouting`, `PowerCabling` |
| `n` | 3 | the three largest `ReadyMix` strength classes, 20,848 to 31,067 values against a corpus ceiling of 9,999 |

The arm's maximum coefficient of variation is 14.34, `PowerCabling`, against a
synthetic maximum of 2.58. Decision 29 records 100 percent coverage; it was
measured on the Stage 2a arm, whose maximum was 2.40. Stage 2a-2 rebuilt the arm
from raw values, the maximum became 13.40, and nothing re-checked.

**The cause is not the draw range**, which reaches 16. The coefficient of
variation is a POPULATION target while the characteristic measured is the SAMPLE
value, which runs low on a right-skewed distribution; Stage 2a-2 recorded that
only 41.7 percent of targets are met. The `n` failures are decision 19 working as
designed, the corpus ceiling of 9,999 with a probe set above it, and splitting
`ReadyMix` turned one uncovered dataset into three.

The five categories failing on dispersion are exactly the ones the author chose
to leave whole as rarely specified: `Aggregates`, `Chairs`, `Elevators`,
`Grouting`, `PowerCabling`. That is worth saying in the paper, because it means
the coverage gap sits on categories the study already flags as not one product.

Not acted on here, deliberately: it is not this stage's scope and acting on it
would mean a fourth regeneration on a question nobody has decided.

**THE DECISION, stated as three options.** Earlier versions of this entry said
"undecided and it needs one" without saying what was on offer, which is a defect
in the document rather than a hard question.

| option | what it means | cost |
|---|---|---|
| **A. Change the text** (recommended) | State coverage as measured and name the exceptions. The claim becomes: the corpus covers the empirical characteristic space with margin except at the extreme upper tail of dispersion, where 5 of 149 datasets sit beyond it, and above 9,999 values per dataset, which the probe set covers by design | nothing; no regeneration |
| B. Widen the generator and regenerate | **MEASURED AND NOT AVAILABLE AS A PARAMETER CHANGE.** Eight candidates were swept in `audits/stage2a3/q5_dispersion_reach.py`: raising the target centre by 0.4, the spread by 1.8x, the upper truncation to 60, and relaxing the quartile-ratio floor `min_q1_over_iqr` from 0.5 through 0.1, 0.05 to 0.01. **The achieved sample coefficient of variation moves from 1.65 to at most 2.15**, against an empirical maximum of 14.34, and NONE of the eight puts a single dataset above 3 | reaching the empirical tail needs heavier-tailed parents or a different truncation rule, which is a generator REDESIGN, not a retune and not one regeneration |
| C. Exclude the uncovered categories | Drops `Aggregates`, `Chairs`, `Elevators`, `Grouting`, `PowerCabling` from the arm | reads the ECC values to decide inclusion, and biases the arm toward low dispersion on the exact dimension the study measures. Advised against |

**Why A is recommended.** The five datasets uncovered on dispersion are exactly
the categories the study already identifies as not one product and leaves whole
for that reason. The exception therefore falls where the paper has already told
the reader to expect trouble, and it can be written as one sentence that
strengthens the account rather than weakening it. The three uncovered on `n` are
decision 19 working as designed: the corpus stops at 9,999 values and the probe
set covers above it.


**Why B fails, measured rather than asserted.** The binding constraint is not the
coefficient-of-variation target. It is the positivity floor of the log truncation
rule, `min_q1_over_iqr`, which caps the parent's quartile ratio at
`1 + 1/min_q1_over_iqr`. At the current 0.5 that is 3, the empirical MEDIAN
quartile ratio, while the five uncovered categories have ratios of 4.5, 27.5,
84.1, 101.3 and 284.6. Relaxing it to 0.01, a cap of 101, still reaches a maximum
sample coefficient of variation of only 2.15. The synthetic datasets are also
nowhere near the arithmetic ceiling of `sqrt(n-1)` that bounds any sample
coefficient of variation, so this is the shape of the parents, not the sample
size.

**A real but partial win, handed to Stage 2h, which already owns
`min_q1_over_iqr`.** Moving it from 0.5 to 0.05 improves the coefficient-of-
variation distribution distance from 0.273 to 0.199 and the visible-mode total
variation from 0.007 to 0.005, with the objective flat at 0.2126 against 0.2118.
It does not close the tail, and this stage did not adopt it, because the roadmap
assigns the parameter to 2h and adopting it would mean a fourth regeneration for
a gain inside the noise. The numbers are in
`outputs/tables/stage2a3/TABLE_2a3_DispersionReach.csv`.

**A correction to section 4.3b.** That section says the matching objective and the
coverage claim pull in opposite directions. For THIS failure they do not: the
empirical arm has 6.0 percent of datasets above a coefficient of variation of 2
and the corpus has 0.08 percent, so closing the gap would improve the match and
the coverage together. The author raised exactly this point. What rules B out is
feasibility, not a conflict of goals.

**Discrepancy entry 34 carries the same table. Raise it before Stage 2b runs
notebook 2.**

### 4.3b Is further retuning worth anything? No.

Four retunes were scored in this stage and **not one moved the objective by as
much as one seed-to-seed standard deviation**, which is 0.0066 from
`audits/stage2a2/p10_config_noise.py`:

| retune | objective | movement |
|---|---|---|
| `cv_log10_sd` 0.3752 to 0.3536, at 440 datasets | 0.2307 to 0.2274 | -0.0033, 0.5 sd |
| `cv_log10_sd` to 0.3919 alone, at 440 | 0.2125 to 0.2186 | +0.0061 WORSE, 0.9 sd |
| `cv_log10_sd` and `cv_log10_mean` together, at 440 | 0.2125 to 0.2118 | -0.0007, 0.1 sd |
| the same pair, full corpus against the resolved arm | 0.2037 to 0.2075 | +0.0038 WORSE, 0.6 sd |

**The tuning has reached its noise floor.** A later stage that reopens generation
to chase the objective will be fitting noise. The reason to have retuned at all
is defensibility, not performance: a parameter must cite a measurement of the arm
actually in use, and `corpus_2026-09-14b`'s did not.

**A deeper question, flagged and not acted on.** The objective matches the SHAPE
of the synthetic characteristic distribution to the empirical one, while the
study also needs to span that space with MARGIN so conclusions generalize past
the categories EC3 happens to hold. In general those can pull apart, because
matching concentrates the corpus where real data is dense. **They do NOT pull
apart on the coverage failure of section 4.3a**: there the corpus is short of the
empirical upper tail on both counts at once, and closing it would improve the
match and the coverage together. What rules that out is feasibility, measured in
4.3a, not a conflict of goals. **Owner: 2f**, which already owns the multivariate
model of where each method wins.

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
| Some EC3 categories are not one product population | 2a-3 | **RESOLVED.** Section 3.1c and decision 46. Three metadata rules, arm 136 to 149 |
| Fold in a newer EC3 pull | 2a-2 | **RESOLVED.** No. Decision 44, the arm is frozen |
| EC3 API is closed to this account | 2a-2 | **RESOLVED.** It is not. Decision 45 |

| Coverage claim is false at the top of the coefficient of variation | manuscript | **RESOLVED as option A**, decision 48. Not an analysis change: the text restates the claim and the coverage figure is rebuilt. Entry 34 |
| A single Dirichlet realization moves per-dataset weighted metrics a long way | 2h | OPEN. Section 4.1, entry 32 |
| Four EC3 parent categories are kept as residual bins because no child is in the arm | - | OPEN as a stated limitation: `PrecastConcrete`, `FireAndSmokeProtection`, `PaintingAndCoating`, `Openings` |
| `CementGrout`, `FlowableFill`, `OilPatch` carry a strength field and are not split | - | OPEN by choice; one tuple in `categorysplit.CONCRETE` changes it |
| EAF against BOF steel is not available | - | CLOSED as infeasible: `steel_making_route_eaf` is populated on 0.01 percent of records |

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

**Stage 2b, the lognormal.** Its first task is unchanged and is now by a wide
margin the oldest outstanding item in the project: **run notebooks 2 and 3
against `corpus_2026-09-14d`**, which `data/processed/CORPUS.json` already points
at, and re-freeze `tests/fixtures/TABLE_EmpiricalECCMetricsAndW1.xlsx` and
`TABLE_SyntheticECCMetricsAndW1.xlsx` in the same commit that moves them. Both
are still pinned to the 136-dataset pre-split arm and will move a long way.

Expect roughly 11 minutes for notebook 3 at `neccs = 10000`. Use
`COMPAREUQ_SMOKE_COMBOS=20` first; CONTEXT.md section 4 says why.

**BOTH INPUTS ARE CLOSED, AND SO IS TUNING.**

- The empirical extract is frozen at the 2026-08 pull, decision 44.
- The category rules are settled, decision 46.
- Generation is closed. It was reopened three times in this stage because the
  arm changed under it; that cannot happen again.
- **No further retuning is warranted**, decision 48. Four retunes here all moved
  the objective by less than one seed-to-seed standard deviation and two made it
  worse. A later stage chasing the objective will be fitting noise. Section 4.3b.

**What Stage 2b inherits that is NOT its job.** The coverage shortfall is a text
edit, decision 48 and entry 34. The Dirichlet weight-realization sensitivity is
2h, entry 32. `min_q1_over_iqr` at 0.05 is a measured partial improvement and is
2h's parameter, section 4.3a.

### Read this before touching the generator or the arm again

The four habits in `reports/HANDOFF_stage-2a2.md` section 7 still hold. This
stage adds two.

5. **A change to the LIST of datasets is a change to every dataset, unless the
   randomness is keyed by identity.** Splitting six categories moved the
   weighted metrics of 130 untouched ones, because the weights were drawn in
   iteration order. Anything drawn per dataset should be keyed by the dataset.
6. **Do not screen material categories on dispersion.** This stage's first
   attempt did, and split on the declared unit, and both were wrong. The
   question is whether a category is one thing a specifier could name, not
   whether it is tight: splitting `ReadyMix` by compressive strength moves its
   coefficient of variation from 0.29 to only 0.27 and is still obviously right.
   Decision 46.
