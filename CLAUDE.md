# CLAUDE.md - Project Brief: CompareUQMethods

This file is read automatically at the start of every session. It carries the
standing project brief so it never has to be pasted again. Later stages append
to it.

---

## Project Brief

I have a Jupyter-notebook analysis for a paper comparing six uncertainty
quantification (UQ) methods for probabilistic LCA of building materials. The six
methods are the cross of three probability estimation methods (normal fit,
lognormal fit, kernel density estimation) with two weighting schemes (uniform,
variable).

### Key vocabulary

- **ECC**: embodied carbon coefficient, the embodied carbon emissions per unit
  mass or volume of a building material, in kg CO2e/kg or kg CO2e/m3.
- **ECC dataset**: the set of ECC values for one building material category,
  drawn from environmental product declarations (EPDs).
- **MUI**: material use intensity, mass of a material per unit floor area.
- **pLCA**: probabilistic LCA. Here, a linear combination of several material
  ECC distributions evaluated by Monte Carlo simulation.
- **W1**: Wasserstein-1 distance, the area between two CDFs. The current
  goodness-of-fit metric.
- **Variable weighting**: weighting each ECC value by the market share of the
  product it represents, instead of weighting all values equally.

### What the analysis currently does

- Extracts 138 empirical ECC datasets from the EC3 database by MasterFormat
  category, cleans them, and normalizes each to a weighted mean of 1.0.
- Computes about 10 statistical characteristics per dataset (coefficient of
  variation, entropy, skewness, kurtosis, mode count, dataset size,
  Shapiro-Wilk normality and lognormality, weight of statistical outliers, and
  W1 between the uniform-weighted and variable-weighted versions of the same
  dataset).
- Generates 10,000 synthetic ECC datasets exhibiting statistical
  characteristics in the ranges observed empirically.
- Assigns variable weights to each data point from a flat Dirichlet
  distribution, since real market share data is unavailable.
- Applies all six UQ methods to every dataset and scores goodness-of-fit as W1
  between each fitted model's CDF and the variable-weighted empirical CDF of
  the data.
- Randomly partitions the 10,000 synthetic datasets into 2,500 disjoint groups
  of four, runs each group as a pLCA by Monte Carlo simulation (n = 10,000,
  MUI = 1.0 for all four) under each of the six UQ methods, and compares
  downstream results, principally "ECI Rank #1 Frequency," the share of Monte
  Carlo iterations in which a given dataset is the largest contributor to the
  total.

The code works but was written by hand over a long period without AI
assistance. It is slow, repetitive, and not organized for reuse. The paper is
drafted and in revision; the code is cited as a public Zenodo deposit. The
target journal is Building and Environment.

### Related work by the author (consistency constraints, not just citations)

Two of my own published papers are directly relevant and should be treated as
constraints on consistency, not just as citations:

- **Torres and Srubar (2025)**, "Characterizing statistical uncertainty and
  variability of building material emissions in probabilistic whole-building
  life cycle assessment using kernel density estimation," Building and
  Environment 284, 113442. This introduced the KDE-based UQ method (**KL1**).
- **Torres, Lupton, Marsh, Srubar and Allen (2026)**, "Using kernel density
  estimation and the Dirichlet distribution for uncertainty quantification of
  building material emissions," Resources, Conservation and Recycling 234,
  109022. This is **KL2**, which adds Dirichlet-sampled market-share weights,
  variable kernel bandwidths for product-level uncertainty, group weight
  constraints, and a representativeness parameter. Code at
  https://doi.org/10.5281/zenodo.19246153

Two papers by Ellen Marsh (University of Bath) are also directly relevant. She
is a co-author on the KL2 paper above and a collaborator from my visiting
appointment at Bath, but is not an author on the present paper:

- **Marsh, Hattam and Allen (2025)**, Journal of Cleaner Production 491,
  144467. Market-share-weighted average LCIA results with uncertainty in both
  the impact data and the production volumes. Useful as a reference for
  realistic weight concentration: in their steel case, Rest-of-World BOF alone
  is 63.75 percent of global production while Austrian EAF is 0.03 percent.
- **Marsh, Lewis, Hattam and Allen (in press)**, "Uncertainty characterisation
  for construction products and comparative metrics for probabilistic building
  LCA." Six uncertainty characterisation scenarios applied to a four-option
  staircase comparison, evaluated with four comparative metrics. Two findings
  matter here: the ranking of top-contributing products within a design changes
  depending on the characterisation scenario, and they use dependent sampling
  across compared options.

Where the current analysis makes a choice that differs from those papers, flag
it rather than silently keeping either version. One is already known: KL2 uses
Silverman's rule for KDE bandwidth and justifies it explicitly, while this
analysis uses Scott's rule.

The manuscript currently cites the KL2 paper as "Torres et al. (in press)." It
is now published, as above. Update every such citation and check for any other
placeholder or in-press citations in the manuscript and code.

### Reference papers

PDFs are in `refs/`. Consult them where noted rather than working from memory;
several define methods this analysis implements directly.

| File | Where it is needed |
|---|---|
| Torres and Srubar (2025), "Characterizing statistical uncertainty and variability of building material emissions in probabilistic whole-building life cycle assessment using kernel density estimation," Building and Environment 284, 113442 | Defines KL1, the KDE method under test |
| Torres et al. (2026), "Using kernel density estimation and the Dirichlet distribution for uncertainty quantification of building material emissions," RC&R 234, 109022 | Defines KL2: Dirichlet weighting, bandwidth with effective sample size and weighted IQR, group weight constraints, representativeness. Stages 2a, 2h |
| Melnykov, Chen and Maitra (2012), "MixSim: An R Package for Simulating Data to Study Performance of Clustering Algorithms," Journal of Statistical Software 51(12) | Overlap parameterization for mixture simulation, with algorithms. Stage 2a Part 5 |
| Marsh, Hattam and Allen (2025), "A method to create weighted-average life cycle impact assessment results for construction products, and enable filtering throughout the design process," JCP 491, 144467 | Real market-share concentration figures. Anchors the concentration sweep in Stage 2h |
| Marsh, Lewis, Hattam and Allen (in press), "Uncertainty characterisation for construction products and comparative metrics for probabilistic building LCA" | Dependent sampling practice and comparative metrics. Stages 2e, 2g |
| Henriksson et al. (2015), "Product carbon footprints and their uncertainties in comparative decision contexts," PLoS One 10(3), e0121221 | Dependent sampling in comparative LCA. Stage 2e |
| Heijungs (2021), "Selecting the best product alternative in a sea of uncertainty," IJLCA 26, 616-632 | Discernibility index and modified comparison index. Stage 2g companion metrics |
| Prado-Lopez et al. (2014), "Stochastic multi-attribute analysis (SMAA) as an interpretation method for comparative life-cycle assessment (LCA)," IJLCA 19, 405-416 | SMAA and overlap area, the main alternative to W1. Implement overlap area alongside W1 as a robustness check in Stage 2c so I can justify the choice in the paper |
| Benke et al. (2025) | Harmonized building LCA quantities. Only if Stage 2i runs |

### Standing constraints for all work on this project

- Plain ASCII in all output, code comments, and figure text. No Unicode
  subscripts, no Unicode multiplication sign, no typographic quotes or dashes.
  Write CO2, not the subscript form.
- Never silently change a result. If a change moves a number, stop and tell me
  which number, by how much, and why.
- Prefer explicit and readable over clever. I need to defend every line of this
  to a reviewer.
- Every analysis writes a tidy results table to disk. Figures are generated
  from those tables, never from in-memory state.
- All randomness comes from an explicitly passed Generator, never from global
  numpy state.

---

## Handoff file specification

Every stage writes a handoff file so the next stage (and the next session)
starts from a written record rather than from memory.

- **Location:** `reports/`
- **Name:** `HANDOFF_stage-<id>.md`, e.g. `reports/HANDOFF_stage-0.md`
- **Format:** plain ASCII Markdown, same constraints as all other output.

Each handoff file contains, in order:

1. **Stage and branch.** Stage id and title, the git branch the work was done
   on, the commit branched from, and the commits made during the stage.
2. **What was asked.** A short restatement of the stage's goal.
3. **What was done.** Findings, decisions, and changes made, in enough detail
   that the work does not have to be redone to understand it.
4. **Numbers that moved.** Any result value that changed during the stage, with
   the before value, the after value, and the reason. States "none" explicitly
   if nothing moved.
5. **Open questions and flags.** Anything deferred, uncertain, or awaiting a
   decision from me, including any divergence from KL1, KL2, or the Marsh
   papers.
6. **Inputs and outputs.** Files read and files written by the stage.
7. **Next stage.** What the next stage should pick up first.

### Continuity across sessions and windows

This project is worked on from several Claude Code windows at once, and
different stages run in different sessions. A session cannot see another
session's conversation. The handoff files are therefore the only channel
between them, and the following rules are binding:

- **Nothing outstanding may live only in a conversation.** Any open question,
  deferred decision, known defect, suspicion, or promise to revisit must be
  written into a handoff file before the stage ends. If it is not in a handoff
  file, it does not exist.
- **Every handoff carries forward the unresolved items from every previous
  handoff**, not only its own. Section 5 of each handoff opens with a
  "Carried forward" list restating every still-open item from earlier stages,
  each marked resolved, still open, or superseded, with the stage that last
  touched it. An item may only leave the list by being marked resolved, with
  the reason given.
- **Start every stage by reading `reports/` in full**, in stage order, before
  doing any work. Do not rely on this file alone; it holds the brief, not the
  state.
- **Record decisions with their reason and their date**, so a later session can
  tell a settled decision from an open one.
- **Never silently reverse an earlier stage's decision.** If a later stage
  finds an earlier decision wrong, say so explicitly in the handoff, name the
  stage and the decision, and state what changed.
- The baseline assessments in `reports/HANDOFF_stage-0.md` section 8 are the
  reference point for any later before-and-after comparison of code quality or
  analysis quality. Do not edit them; later stages record their own assessment
  in their own handoff.

---

## Repository conventions

- Branch per stage: `stage-<id>-<short-name>`, e.g. `stage-0-1-refactor`.
- Commit in logical units so any result change can be bisected.
- `refs/` is gitignored: it holds copyrighted publisher PDFs and large
  third-party datasets, kept locally only.
