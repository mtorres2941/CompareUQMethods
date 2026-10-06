# CompareUQMethods

**Uncertainty quantification methods for probabilistic whole-building life cycle assessment: A comparative analysis**

Martín I. Torres, Wil V. Srubar III
University of Colorado Boulder

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19226429.svg)](https://doi.org/10.5281/zenodo.19226429)

---

## Overview

This repository contains the data, source code, and analysis notebooks supporting the paper above. The study evaluates six combinations of probability estimation (PE) and weighting (WT) methods, referred to as PEWT methods, to determine which best represents the distribution of environmental carbon footprint data used in probabilistic whole-building life cycle assessment (pWBLCA).

The three PE methods compared are:

| Label | Method |
|---|---|
| **Normal** | Gaussian fit, truncated to (0, inf) and renormalized |
| **Lognormal** | Three-parameter lognormal, threshold chosen by profile likelihood under a scale-aware guard |
| **KDE** | Kernel density estimation at Silverman's robust bandwidth, guarded by a minimum effective sample size |

Each PE method is evaluated under two WT strategies:

| Label | Strategy |
|---|---|
| **Uniform** | Equal weights across all data points, which is what a practitioner pulling declarations from a database actually holds |
| **Variable** | Market-share weights. `Variable` is the stored label, kept because every table and fixture in the project joins on it; everywhere a reader sees it, the scheme is called **market weights** |

**"Market weights" does not mean real production volumes, and the paper says so
at first use.** Nobody publishes product-level market shares, so the shares are
simulated. What the two arms do with them differs, and the difference is the
point of the comparison: on the **synthetic** arm the share attached to each
product group is the group's TRUE share in the parent the data were drawn from,
exact to 1.1e-16, and only its division among the products inside the group is
arbitrary -- so uniform against market there is IGNORING a known share against
USING it. On the **empirical** arm there is no truth to know, so the shares are
simulated under a published-volume-shaped model and the arm says what weighting
WOULD do, not what ignoring a known share costs.

Performance is measured as the Wasserstein-1 distance between each fitted model
and the distribution it is estimating -- the known parent for the synthetic
datasets, a cross-validated held-out sample for the real ones -- and by the
error each method causes in the answer a probabilistic LCA gives, measured
against the true distributions the synthetic data were drawn from.

---

## Repository structure

```
CompareUQMethods/
|-- notebooks/                         # Analysis notebooks (run in order)
|   |-- 01_CompareUQ_CreateData.ipynb
|   |-- 02_CompareUQ_AnalyzeData.ipynb
|   |-- 03_CompareUQ_PerformPLCA.ipynb
|   `-- 04_CompareUQ_ReduceMetrics.ipynb
|-- src/                               # Tested Python library the notebooks call
|-- audits/                            # One-off measurements cited by the decision log
|-- tests/                             # pytest suite over src/ and the notebooks
|-- data/
|   |-- raw/                           # Frozen, checksummed EC3 extracts
|   `-- processed/                     # Corpus pointer, the corpora, and the
|                                      # prepared empirical arm. See its README
|                                      # for WHICH corpus the paper describes
|-- outputs/
|   |-- figures/                       # Publication figures, PNG at 300 dpi
|   |                                  # with a PDF sibling for each
|   `-- tables/                        # Result tables (CSV, XLSX, Parquet)
|-- archive/                           # Superseded figures, each with the
|                                      # reason it was replaced
|-- reports/                           # The stage prompts, the manuscript
|                                      # discrepancy log, and timing baselines
|-- CLAUDE.md                          # Project brief and the full decision log
|-- CONTEXT.md                         # Package layout, conventions, test inventory
|-- FIGURE_STYLE.md                    # Binding style guide for every figure
`-- LICENSE
```

**The paper describes ONE synthetic corpus, `corpus_2026-09-25`.**
`data/processed/CORPUS.json` names it and the notebooks read it from there.
The other `corpus_*` directories beside it are superseded, and they are kept
because a corpus here is immutable -- a change to generation writes a new
directory so the two can be diffed file by file, which is how a change is
proved to have moved only what it was meant to. They are not a second dataset
and no number in the paper comes from any of them. `data/processed/README.md`
says this in full, and says which files are tracked and how to rebuild the
large ones.

**Every figure and table under `outputs/` is reproducible from a notebook cell,
and the code that produces it lives in a notebook.** Nothing reaches `outputs/`
whose source a reader cannot open and read there. A different EXECUTOR of a
notebook's own cells is allowed -- `audits/render_figures.py` re-runs the
notebook's figure cells verbatim so a figure can be iterated on in seconds
instead of in a notebook run -- and a different AUTHOR is not. Audit scripts
write only to `outputs/tables/audits/`.

`python audits/figure_manifest.py` lists every image with the code that writes
it, and fails the test suite on an image with no generator or a filename two
places write.

---

## Workflow

The analysis runs as four sequential notebooks. Notebooks 2, 3 and 4 depend on
artifacts written by the ones before them.

### Notebook 1 - Create data (`01_CompareUQ_CreateData.ipynb`)

**Empirical datasets.** Reads a frozen, dated raw extract of the EC3 database,
resolves EC3 categories into specifiable products on record metadata only
(residual bins dropped, concrete split by specified compressive strength,
insulation by material type), removes physically implausible records against an
external bound, and cleans each category with a symmetric log-space 3 x IQR
rule. The result is **147 empirical ECC datasets**, each normalized to an
unweighted mean of 1.0.

**Synthetic datasets.** Generates **10,000 synthetic ECC datasets** from
moment-targeted mixtures of Johnson-system and Pearson components, stratified
by size over 3-9, 10-99, 100-999 and 1000-9999 with 2,500 in each stratum.
Component overlap, the coefficient of variation and the visible-mode
distribution are specified generation targets solved for per dataset, and the
configuration is tuned so the synthetic characteristic distributions match the
empirical ones. A corpus is a named, dated, immutable directory carrying its
seed, configuration, git commit and library versions.

**Weights.** Both arms draw market shares by the **same rule**, which they did
not until late in the project: declarations are divided into product groups,
each group's share of the market is drawn from a flat Dirichlet, and that share
is divided among the declarations inside the group. What differs is how the
groups are found, and it has to. The synthetic arm KNOWS them -- they are the
mixture components the data were drawn from -- so the share it attaches to a
group is that group's true share in the parent. The empirical arm cannot know
them, so `weighting.coherent_weights` cuts the sorted values into contiguous
groups instead, at a coherence of 0.5: the value at which that cut reproduces
what the synthetic arm's TRUE component labels give, to within four percent on
the typical dataset. It is measured against the truth, not chosen.

Drawing a flat Dirichlet over every declaration independently, which is what
the empirical arm did before, is not the neutral alternative: it is the
specific claim that market share is uncorrelated with carbon intensity, which
published production volumes contradict. It also forces the measured weighting
effect to decay as the dataset grows, which is a property of that model and not
of markets.

**Characteristics and weighting risk.** Computes the statistical
characteristics of every dataset under both weightings, and the per-dataset
probability that assuming uniform weights changes which material ranks first.

### Notebook 2 - Analyze data (`02_CompareUQ_AnalyzeData.ipynb`)

Fits all six PEWT models to every dataset and scores goodness-of-fit as the
Wasserstein-1 distance between each fitted CDF and its target, on a
20,000-point trapezoid grid plus an analytic tail term.

**The evaluation target differs by arm, deliberately.** The synthetic arm is
scored against the **known parent distribution** the data were drawn from. The
empirical arm, which has no parent, is scored by **cross-validated** W1 over
random half-splits shared by all six methods. Scoring a model against the same
weighted data it was fitted to is circular twice over, and the in-sample score
is retained only for comparison.

Every headline aggregate is reported twice: under equal allocation across size
strata, and reweighted to the empirical size mix.

### Notebook 3 - Perform pLCA (`03_CompareUQ_PerformPLCA.ipynb`)

Partitions the corpus into **2,500 groups of four materials** and runs each as a
probabilistic LCA by Monte Carlo simulation at 10,000 draws under each of the
six PEWT methods, using **common random numbers** so that two identical models
give identical results and the difference between methods carries no sampling
noise.

Reports what the choice of method does to the building total, to each
material's contribution and share, to where the uncertainty sits, and to the
decisions a designer makes: capping a specification, reducing a quantity, and
swapping one material for another. Every pLCA is also run against the datasets'
**true parent distributions**, so the error each method causes is measured
directly rather than inferred.

### Notebook 4 - Reduce metrics (`04_CompareUQ_ReduceMetrics.ipynb`)

Asks which of the ~25 statistical characteristics actually decide **which
method to use**, and reduces the set to the handful that carry independent
signal.

The target is the difference between two methods rather than the level of one,
because the level is dominated by dispersion and dataset size for every method
alike. Gains are **cross-validated** and reported beside their spread across
folds rather than against a p-value, and **forward selection** admits a
characteristic only if it still helps once everything already chosen is in the
model, so correlated views of one shape are not counted several times.

**Claims rest on the 10,000 synthetic datasets, with the 147 real categories as
a consistency check.** That is what the synthetic corpus is for: 127 scorable
real categories cannot support this model out of sample.

Produces the practitioner-facing thresholds: the dataset size above which a
kernel estimate is the better default, the cost of getting that threshold
wrong, and when market-share weighting pays.

---

## Installation

**Requirements:** Python 3.11, and a conda environment pinned by
`environment.yml`.

```bash
conda env create -f environment.yml
conda activate compareuq
pip install -e .
```

---

## Usage

Run the notebooks in order from the `notebooks/` directory:

```
01_CompareUQ_CreateData.ipynb    ->  data/processed/, outputs/
02_CompareUQ_AnalyzeData.ipynb   ->  outputs/figures/, outputs/tables/
03_CompareUQ_PerformPLCA.ipynb   ->  outputs/figures/, outputs/tables/
04_CompareUQ_ReduceMetrics.ipynb ->  outputs/figures/, outputs/tables/
```

Approximate wall-clock time on a 2026 laptop with nothing else running:
notebook 1 about 35 minutes, notebook 2 about 15, notebook 3 about 110 minutes,
notebook 4 about 15. A machine running other work can take half again as long,
which is why these are quoted to the nearest five minutes.

**Notebook 3 is the expensive one and the only one with a smoke configuration**:
`COMPAREUQ_SMOKE_COMBOS=20` runs 20 probabilistic LCAs instead of 2,500 and
redirects every write to a temporary directory, so the pipeline can be exercised
end to end in under a minute without touching `outputs/`. Use it before any full
run.

**To change a figure without re-running anything**, use the renderer:

```
python audits/render_figures.py 02_CompareUQ_AnalyzeData --only "regret"
python audits/render_figures.py 02_CompareUQ_AnalyzeData --into-outputs
```

The tests run with `pytest` from the repository root and cover the source
library and the notebooks themselves, including that every code cell parses,
that no cell reads a frame a later cell defines, that every figure cell can be
redrawn on its own, and that all randomness comes from an explicitly passed
Generator.

**What each output table holds** is in `CONTEXT.md` section 6, which lists
every file under `outputs/tables/` with the notebook that writes it and its
shape, and marks the handful a reader should start from. The figures are in
the manifest below.

---

## Figure manifest

Every image under `outputs/figures/` with the notebook that writes it and what
it shows. Each file has a `.pdf` sibling of the same name; names are given here
without the `CompareUQMethods_` prefix and the extension.

**The `FIG_` and `SUPP_` prefixes record what the generating cell declares
itself to be, not where the manuscript puts it.** The manuscript's figure
selection and numbering are settled while the manuscript is written, which is
after this deposit is cut, so no file here carries a figure number. Renaming is
one word per cell -- `figstyle.savefig` takes a stem, not a path -- and
`tests/test_figure_manifest.py` fails on any file a rename would leave behind
without a generator, so the renumbering is cheap whenever it happens.

`python audits/figure_manifest.py` regenerates this list from the code.

### Notebook 1 - the data

| File | What it shows |
|---|---|
| `FIG_DemonstrateDataGeneration` | How one synthetic dataset is built, from drawn targets to realized sample |
| `SUPP_DatasetExamplesByStratum` | Synthetic datasets by size stratum, with real categories beside them |
| `FIG_MetricCoverage` | Where the 147 real categories sit inside the synthetic cloud, characteristic by characteristic |
| `SUPP_GeneratedVsEmpiricalMetrics` | Every statistical characteristic, the two arms' distributions overlaid |
| `FIG_WeightingDrivers` | Which categories can safely assume uniform weights, against size and dispersion |
| `FIG_ShapePlane` | Why a two-parameter lognormal cannot fit this data: its skewness is fixed at CV^3 + 3 CV, and only 27 of 127 real categories sit on that curve |
| `FIG_WeightingBySize` | When knowing a market share starts to help, against dataset size, with the real categories' own sizes underneath |

### Notebook 2 - the fits

| File | What it shows |
|---|---|
| `DEF_DemoW1Dist` | What a Wasserstein-1 distance is: the area between two cumulative curves |
| `FIG_PDFandCDFofUQMethods` | The six UQ methods on one dataset, as densities and as distribution functions |
| `FIG_W1DistanceAndRank` | How far each method sits from its target, and how often it is closest. Both arms |
| `SUPP_KSTestStripAndRank`, `SUPP_Wass2DistStripAndRank` | The same under two other distances, as a robustness check |
| `FIG_W1VsCharacteristic_{Empirical,Synthetic}` | W1 against every dataset characteristic, one panel each |
| `FIG_W1VsSurvivors_{Empirical,Synthetic}` | The same for the characteristics the reduction keeps, the two weightings side by side |
| `FIG_WeightingGap_{Empirical,Synthetic}` | The alternative to the above: the uniform-minus-market difference, three curves instead of six |
| `SUPP_ByCharacteristic_<name>` (12 files) | One page per characteristic, so each can be read on its own |
| `FIG_RankVsDatasetSize` | Which method wins against dataset size, and by how much |
| `FIG_WinShareVsCharacteristic_{Empirical,Synthetic}` | Which method wins, by percentile of each characteristic |
| `FIG_BandwidthRule` | What the bandwidth rule does, against dataset size |
| `FIG_EvaluationTarget` | What changing the scoring target does to the comparison |
| `FIG_TargetBySize` | The size dependence, which is what both arms agree on |
| `FIG_Regret` | What it costs to use one method on every dataset instead of the best one for each |
| `FIG_MethodByMaterial` | Which default to use, and what it costs, by material tier |
| `SUPP_AllEmpiricalFits` | Every real category with all six fitted models |

### Notebook 3 - the probabilistic LCA

| File | What it shows |
|---|---|
| `DEF_VisualizeReductionStrategies` | The four reduction strategies, drawn |
| `FIG_ScatterPlot_UQResults_Subset` | What switching method does to four pLCA outputs, every pair of methods on every pLCA |
| `SUPP_ScatterPlot_UQResults_All` | The same for every output |
| `SUPP_AllResultsByAllUQMethods` | The NRMSE between every pair of methods, for every output |
| `FIG_WassVsResultDiff` | The distance between two fitted models against the difference it makes downstream |
| `FIG_PLCAW1Distances` | W1 between the six methods, for the three example pLCAs |
| `FIG_PLCAVisualizeUQFits` | The components and the total of those three pLCAs |
| `FIG_RanksByDatasetAndPEWT` | How often each material leads, by dataset and method, in those three |
| `FIG_FlipCalibration` | How far apart two models must be before the answer changes, and what such a distance looks like |
| `FIG_MaterialDominance` | What a leading material buys -- the ranking -- and what it does not -- the magnitude |
| `FIG_PLCATruth` | How wrong each method's answer is against the true parents, as a distribution |
| `FIG_ClaimScorecard` | Every claim a probabilistic LCA makes, scored for the six methods and the size rule |
| `FIG_MixedPolicy` | How much the cutoff matters, and what knowing market share would buy |
| `FIG_BuildingDominance` | Where 292 real North American buildings sit on the safe-lead axis (Benke et al. 2025) |

### Notebook 4 - which characteristics decide

| File | What it shows |
|---|---|
| `FIG_WhenToUseWhich` | Which method is closest to the truth, against category size |
| `FIG_ChoiceDrivers` | Whether dataset size is enough on its own |
| `SUPP_RollingVersusBinned_{Empirical,Synthetic}` | The rolling average against its replacement, kept so the two can be compared |

---

## Source modules

| File | Description |
|---|---|
| `src/empirical.py` | Reads the frozen EC3 extract, applies the plausibility bounds and the log-space cleaning, and prepares the empirical arm |
| `src/categorysplit.py` | Resolves EC3 categories into specifiable products, on record metadata only and provably without reading any ECC value |
| `src/genconfig.py`, `src/generator.py`, `src/components.py`, `src/mixture.py` | Synthetic dataset generation: the configuration, the per-dataset solve, the moment-targeted components and the truncated mixture parent |
| `src/corpus.py` | Writes, loads, re-measures and replays a named corpus; `remetric_corpus` recomputes characteristics without redrawing a single value |
| `src/customstats.py` | Weighted statistics: ECDF, quantiles, bandwidth rules, Shapiro-Francia, the modality index, the characteristic set |
| `src/modality.py` | Silverman's critical bandwidth and the visible-mode counts |
| `src/families.py` | The three probability families as explicit truncations to (0, inf), each exposing pdf, cdf, ppf and inverse-CDF sampling |
| `src/fitting.py` | The single fitting and scoring implementation, including the W1 grid, route and tail term |
| `src/recovery.py` | Scores a fitted model against a known parent, and post-stratifies |
| `src/comparison.py` | Cross-validated scoring and the paired bootstraps |
| `src/weighting.py` | The one market-share rule both arms use, the uniform-to-market separation with its location and shape split, and the weighting-risk measures |
| `src/flip.py` | Calibrates how far apart two models must be before the answer changes |
| `src/plca.py` | The probabilistic LCA, common random numbers, the interventions and the design swap |
| `src/metricreduction.py` | The metric reduction: cross-validated gains, forward selection, permutation importance, and the practitioner thresholds |
| `src/materialclass.py` | Assigns each category a material tier from its name alone |
| `src/coverage.py` | Compares the two arms' characteristic distributions |
| `src/figstyle.py` | Implements what `FIGURE_STYLE.md` can implement, including the label-overlap and ink checks |
| `src/funcs_unit_conversion.py` | Unit conversion for normalizing EC3 records |

---

## Citation

If you use this code or data in your research, please cite:

> Torres, M.I. and Srubar III, W.V. (submitted). Uncertainty quantification methods for probabilistic whole-building life cycle assessment: A comparative analysis. *Building & Environment*.

The dataset and code are archived on Zenodo: https://doi.org/10.5281/zenodo.19226429

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
