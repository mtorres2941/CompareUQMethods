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
| **Variable** | Market-share weights, drawn from a flat Dirichlet distribution because real production volumes are not published |

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
|   `-- processed/                     # Corpus pointer and prepared empirical arm
|-- outputs/
|   |-- figures/                       # Publication figures, PNG at 300 dpi
|   |                                  # with a PDF sibling for each
|   `-- tables/                        # Result tables (CSV, XLSX, Parquet)
|-- reports/                           # Stage handoff and manuscript discrepancies
|-- CLAUDE.md                          # Project brief and the full decision log
|-- CONTEXT.md                         # Package layout, conventions, test inventory
|-- FIGURE_STYLE.md                    # Binding style guide for every figure
`-- LICENSE
```

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

**Weights.** Both arms receive market-share weights drawn from a flat Dirichlet,
since real production volumes are not published. On the synthetic arm share
attaches at the mode level, so the market-weighted distribution is a real
population object rather than a property of one realized sample.

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
notebook 4 about 20. A machine running other work can take half again as long,
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
that no cell reads a frame a later cell defines, and that all randomness comes
from an explicitly passed Generator.

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
| `src/weighting.py` | The uniform-to-variable separation, its location and shape split, and the weighting-risk measures |
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
