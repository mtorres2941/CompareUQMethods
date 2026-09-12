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

## Pipeline roadmap

Everything that is queued, so that a session can recognize an item belonging to
a later stage and leave it alone instead of either solving it early or worrying
that it has been forgotten.

**How to use this list.** If you hit something this list assigns to another
stage, do not fix it, do not sketch a solution, and do not run a quick check
"just to know." Write one line in your handoff under the carried-forward items
saying what you saw and which stage owns it, and carry on with your own stage.
The failure mode this list exists to prevent is a stage that does its own work
plus a thin version of three others, leaving the author unable to tell which
number came from which decision.

**The one hard sequencing rule:** regeneration of the synthetic datasets
happens exactly once, at the end of Stage 2a. Every number in the paper moves
when it does. Any stage that touches generation parameters before 2a closes, or
regenerates again afterward, doubles the verification work for no gain.

| Stage | Owns | Explicitly not its job |
|---|---|---|
| **0 DONE** | Inventory, dependency map, refactor plan, baseline code and analysis assessments. `reports/HANDOFF_stage-0.md` | Any change to analysis logic |
| **1 DONE** | Pinned environment, regression fixtures, persisted pLCA table, seeding machinery, correctness fixes, thin notebooks over a tested `src/`. Exactly two intended number-moving changes: neccs 1,000 to 10,000, and the wbeci assignment moved inside the loop. `reports/HANDOFF_stage-1.md` | Regeneration, and every methodological judgment call. Amendment A3 settled normalization: unweighted mean, code stands, text is wrong |
| **2a DONE** | Generator audit: seeding collapse, Dirichlet concentration mismatch, stale docstring, the truncation loop, power transform, reflection, component overlap, mode counting, the 27.5 percent filter, mode-level market share, the coverage table that becomes Table 1. Then regenerate, once | Changing the fitting methods, changing the scoring target, or sweeping anything that 2h owns |
| **2b** | The lognormal: threshold pathology, the +0.5 offset, two-parameter versus profile-likelihood versus gamma. W1-optimal fitting alongside MLE | Adding new families for robustness (2h), or rescoring against a parent (2c) |
| **2c** | The evaluation target: score synthetic against the known parent, cross-validate the empirical 138, decompose location versus definitional error, report regret distributions. Overlap area alongside W1 | The pLCA construction (2e) and the flip-probability threshold (2d) |
| **2d** | Decompose the uniform-to-variable W1 into location and shape, define the named relative measure, calibrate flip probability against relative W1, report the 1, 5 and 10 percent crossings | Building companion decision metrics (2g) |
| **2e** | pLCA construction: common random numbers across UQ methods, sweep materials per pLCA over 2 to 12, resample groupings, dominant-MUI variant, bootstrap intervals on every headline percentage and NRMSE | Changing what the headline metric is (2g) |
| **2f** | Resolve Shapiro-Wilk versus Shapiro-Francia and `_royston_pvalue`, then the multivariate model of W1 and of which method wins, to cut the metric set to three to five survivors | Regenerating, or redesigning figures (3) |
| **2g** | Sensitivity of ECI Rank #1 Frequency, magnitude-based companions, and the `(1-capecc)` divisor | Re-running the sweeps of 2h |
| **2h** | Robustness sweeps: KDE bandwidth (Scott, Silverman with a degenerate-IQR guard, cross-validated), lognormal offset, gamma and Weibull as extra families, Dirichlet concentration, multiple weight realizations, mode-to-point coupling | Anything not framed as a sweep with a tabulated result |
| **2i** (optional) | Real-building anchor, only if we decide after 2g that citing Marsh et al. (in press) is not enough | Becoming a case study |
| **3** | Figures: merge 2 and 3, rebuild 4 from the 2f survivors, the figure manifest, the naming convention, vector output, duplicate-filename check. **Also the figure SIZE problem: several figures declare a `figsize` of roughly 94 by 55 inches and come out at 50 to 98 megapixels. 2a deleted 15 stale figures (115 MB) but did not touch the live ones, which are all from the pre-regeneration corpus and will be rebuilt anyway** | Changing any number |
| **4** (optional) | README and Zenodo re-deposit | Anything analytical. **NOT the `.git` history rewrite: declined by the author, decision 28** |

Items already known to be open and owned by a named stage, so that none of them
reads as a fresh discovery: the bandwidth rule and its KL1/KL2 inconsistency
(2h); `logfit_offset` (2b, swept in 2h); the "Mode Count" label naming a
continuous modality index (2a); the 27.5 percent filter and its n cap at 749
(2a); the variance-inflation exponent and the reflection step (2a);
Shapiro-Wilk versus Shapiro-Francia and `_royston_pvalue` (2f); dependent
sampling (2e); overlap area alongside W1 (2c); the `(1-capecc)` divisor (2g);
`weighted_quantile` order dependence (fixed in Stage 1 Phase 3, and it must
stay fixed before any switch to Silverman in 2h).

Mark each stage done as it completes. If a stage hands an item to a different
stage than this table says, update the table rather than leaving the two out of
step.

**Outside the stage structure, and belonging to the manuscript session, not to
any analysis stage:** revising the manuscript, and
`reports/MANUSCRIPT_discrepancies.md`, which every stage appends to and which
is worked from later. Keep appending. Do not start editing the paper.

---

## How the repository works

Mechanics live in **CONTEXT.md**: package layout, the fitting interface, the
seeding and caching conventions, how to run the pinned environment and the
smoke configuration, the input and output tables, the regression fixture
inventory and what each one pins, and the test suite. Read it before touching
code. It was split out of this file at the end of Stage 1, when this file grew
past a comfortable size.

---

## Decision log

Newest last. Every entry gives the decision, the date, and the reason. A later
stage must never silently reverse one of these; if it finds a decision wrong,
it says so explicitly in its handoff, naming the decision and what changed.

**Provenance tags.** Entries 10 onward are tagged, because several arose in
conversation and a later window cannot see that conversation. The distinction
matters: a **[AUTHOR]** item is settled and a later stage should implement it;
a **[RECOMMENDED]** item is a Stage 1 suggestion the owning stage is free to
overrule; a **[DELEGATED]** item is one the author explicitly left to
judgment. Untagged entries 1 to 9 are all author decisions, taken in a prompt
rather than in conversation.

1. **2026-09-11, Stage 0. `refs/` is not tracked.** It holds copyrighted
   publisher PDFs and a 104 MB third-party dataset. This repository is public
   and Zenodo-archived.
2. **2026-09-11, Stage 0. Manuscript drafts are not tracked.** The working
   `.docx` carries 98 unresolved comments from named third parties, and git
   history would retain them even after deletion.
3. **2026-09-11, Stage 0. Notebooks remain the entry point.** The author values
   seeing inputs and outputs inline and considers notebooks more reviewable by
   an outside reader. The Stage 0 session initially recommended scripts and
   withdrew it: the defects found came from leaked kernel state, duplicated
   logic and untested code, not from notebooks as a medium.
4. **2026-09-11, Stage 0. The synthetic datasets are regenerated once, in
   Stage 2.** Several Stage 2 decisions concern the generation algorithm
   itself, so regenerating in Stage 1 would mean regenerating twice and
   invalidating the paper's numbers twice.
5. **2026-09-11, Stage 0. Invalidating the manuscript's numbers is accepted.**
   The analysis is being redone because of structural problems. No
   number-preservation constraint applies to the final results; the "never
   silently change a result" constraint still applies in full, meaning every
   change must be attributable, recorded and intentional.
6. **2026-09-11, Stage 1 (A3). Normalization stays on the unweighted mean.**
   `data / np.mean(data)` in both the synthetic and empirical paths. The
   practitioner-facing threshold this study builds must be computable by
   someone who has a set of EPDs but does not know the market shares, which is
   precisely the quantity they lack. The manuscript text is what is wrong; see
   `reports/MANUSCRIPT_discrepancies.md` entry 2.
7. **2026-09-11, Stage 1 (A2). Three pLCA artifacts are kept**: the unseeded
   archive at neccs=1000, the seeded run at neccs=1000, and the seeded run at
   neccs=10000. The first is unreproducible and is the closest surviving record
   of what the current manuscript reports.
8. **2026-09-11, Stage 1. The pLCA runs at 10,000 draws.** The manuscript
   states this throughout, and at 1,000 the Monte Carlo noise in `eci_rank_1`
   was 15.5% of that result's standard deviation across datasets, leaving the
   weakest pair of UQ methods only 2.2x above the noise floor.
9. **2026-09-11, Stage 1. Bandwidth labels are correct as written.**
   `weighted_bw`'s `'scott'` is Scott (1992) and its `'silverman'` is
   Silverman's robust rule of thumb. The hazard is that `scipy.stats.gaussian_kde`
   uses the same two words for different formulas. An earlier Stage 0 note
   wrongly implied the project's labels were wrong.
10. **2026-09-11, Stage 1. The lognormal keeps a threshold.** Owned by 2b.
    - **[AUTHOR]** Use a three-parameter lognormal, and do not constrain the
      threshold to stop it approaching the normal limit: "that sounds like an
      asset of the lognormal distribution. It can accommodate more datasets.
      That's a good thing, not a bug... Why unnecessarily cripple it?"
    - **[CONTEXT]** The near-zero pathology `LOGFIT_OFFSET` was patching is
      real, and worse in the empirical data than the synthetic: 28.3% of the
      138 empirical datasets have a minimum below 1% of their mean, against
      1.8% of the synthetic. A Stage 1 recommendation to simply drop the offset
      was withdrawn as wrong.
    - **[RECOMMENDED]** State parameter counts plainly and consider a held-out
      or cross-validated W1, because W1 is an in-sample criterion with no
      complexity penalty and the three families differ in flexibility. 2b and
      2c may overrule this.
11. **2026-09-11, Stage 1. Multimodality.** Owned by 2a.
    - **[AUTHOR]** The question of interest is unimodal versus multimodal.
      "I don't want to distinguish bimodal from trimodal." Whatever measure is
      chosen, report the test statistic and not a p-value: the author is
      "highly skeptical of p-values" and prefers statistics directly.
    - **[RECOMMENDED]** Hartigan's dip statistic. The author said they were
      "open to using" it, not that it is settled. It needs no bandwidth, which
      also avoids using KDE to justify KDE, and `n` spans 4 to 10,000 so a
      p-value would conflate effect size with sample size. 2a chooses.
12. **2026-09-11, Stage 1. Cleaning.** Owned by 2a.
    - **[AUTHOR]** The cleaning must be fixed: "I agree with fixing the
      cleaning." The multiplicative idea was the author's own, offered as
      "maybe we should take a multiplicative approach", so the direction is
      theirs but the specific filter is not settled.
    - **[CONTEXT]** The additive `Q1 - 3*IQR` bound is negative in 128 of 138
      empirical datasets, so it never binds: near-zero values are never removed
      while high outliers are. `ReadyMix` retains a value at 3.1e-17 of its
      mean, which is a data error, not a product.
    - **[RECOMMENDED]** IQR in log space. On ReadyMix it keeps 77,439 of 77,548
      values with bounds [90.6, 1240]. 2a chooses the final form, including
      whether synthetic generation needs a matching floor.
13. **2026-09-11, Stage 1. Support is (0, inf), open at zero.**
    - **[AUTHOR]** "I think it's (0, infinity), rather than [0, infinity) since
      we shouldn't accept zero as an input."
    - **Reach, flagged for confirmation.** This constrains the lognormal and
      gamma fits in 2b and the W1 evaluation grid in 2c. It was stated in
      conversation rather than in a prompt, so 2b or 2c should confirm it with
      the author before building on it. The pLCA rejection sampling already
      enforces it; the scoring grid starting at exactly 0 does not.
14. **2026-09-11, Stage 1. Dataset size range and the n filter.** Owned by 2a.
    - **[AUTHOR]** "I like the idea of extending to 10,000 for the sample size.
      I also agree with exempting n from the outlier filter."
    - **[CONTEXT]** The filter discards 824 datasets for being large, capping
      the effective maximum at 749 against a stated 1,000. Empirical sizes
      reach 77,548. Extending to 10,000 covers 132 of 138 empirical datasets;
      going to 77,548 would put 44% of synthetic datasets above n=1,000, which
      is unrepresentative of an empirical median of 37, and would make
      `DATA_all` roughly 10 GB.
15. **2026-09-11, Stage 1. `DATA_all` storage format.** Owned by 2a.
    - **[AUTHOR]** Move off JSON. "That was just because I know JSON better
      than other file types."
    - **[DELEGATED]** The format itself: "Please go with what you think is
      best."
    - **[RECOMMENDED]** Parquet in long format (`dataset_id, value, weight`):
      columnar, compresses roughly 19M rows to a few hundred MB, and readable
      from R and Julia as well as Python, which matters for a Zenodo deposit.
      Not `.npz`, which is smaller but opaque outside Python.

---

## Repository conventions

- Branch per stage: `stage-<id>-<short-name>`, e.g. `stage-0-1-refactor`.
- Commit in logical units so any result change can be bisected.
- `refs/` is gitignored: it holds copyrighted publisher PDFs and large
  third-party datasets, kept locally only.

16. **2026-09-11, Stage 2a. Dirichlet concentration is alpha = 1 in both
    arms.** `[AUTHOR]` The empirical arm used 5 and the synthetic arm 1, on the
    exact dimension the paper is about. Correcting the empirical arm more than
    doubles the mean uniform-to-variable W1 across the 138 datasets, 0.0594 to
    0.1329. Note for the manuscript: alpha is the Dirichlet CONCENTRATION
    parameter, so a smaller alpha gives MORE dispersed market shares and the
    measured weighting effect goes UP. It is still a lower bound on reality, as
    a flat Dirichlet at n = 100 gives an expected top share of 5.2 percent
    against the 63.75 percent Marsh, Hattam and Allen (2025) report for
    Rest-of-World BOF steel.
17. **2026-09-11, Stage 2a. The +1 buffer is removed.** `[AUTHOR]` Undocumented,
    asymmetric between the arms, and it compressed the coefficient of
    variation in 28.8 percent of datasets by more than 1 percent.
18. **2026-09-11, Stage 2a. The truncation loop, the power transform and the
    reflection are removed.** `[AUTHOR]` Sampling is now direct inverse-CDF
    from a mixture truncated at POPULATION quantiles. Skewness is a component
    moment target and left skew comes from reflected one-sided families, so
    the parent CDF is closed form. Verified against 400,000 draws.
19. **2026-09-11, Stage 2a. Dataset size is stratified**, 2,500 each over 3-9,
    10-99, 100-999 and 1000-9999, with a 50-dataset probe set at 10,000 to
    100,000 held outside every aggregate. `[AUTHOR]`
20. **2026-09-11, Stage 2a. The 27.5 percent metric-outlier filter is
    dropped**, replaced by a validity-only filter. `[AUTHOR]` It preferentially
    removed high weight_outliers datasets (0.83 pooled sd difference), capped
    n at 749 via 824 removals on size alone, and flagged 14 datasets on a
    column that is 1.0 by construction.
21. **2026-09-11, Stage 2a. Market share attaches at the mode level**, with a
    coupling parameter that reduces to the old uncoupled behaviour at 0.
    `[AUTHOR]` Both weighting schemes now have a population to be right or
    wrong about.
22. **2026-09-11, Stage 2a. Components are moment-targeted Johnson-system and
    Pearson families.** `[DELEGATED, 2a chose]` Johnson SU above the lognormal
    line, lognormal on it, beta-prime between the gamma and lognormal lines,
    beta below the gamma line. Each has a closed-form CDF and exact moments, so
    a component is specified by (mean, sd, skewness, kurtosis) and solved for.
    Targets that cannot be met are reported, never approximated.
23. **2026-09-11, Stage 2a. Modality is Silverman's critical bandwidth**, not
    Hartigan's dip. `[DELEGATED, 2a chose]` Decision 11 left the choice to 2a
    and required a statistic rather than a p-value; `crit_bw_1` is reported in
    units of the data's standard deviation and the bootstrap level is kept as a
    secondary diagnostic. `mode_count_est` is renamed `modality_index`, which
    is what it measures: across the 138 empirical datasets it spans only 1.000
    to 1.159, so as a count it is constant at 1, while Silverman finds 27 of
    them multimodal.
24. **2026-09-11, Stage 2a. Empirical cleaning gains a multiplicative low-end
    bound only.** `[DELEGATED, 2a chose]` Decision 12 left the form to 2a. The
    high end was already trimmed additively when the surviving file was
    written and the EC3 source directory no longer exists, so a symmetric
    re-clean would trim the same tail twice. Removes 342 of 107,523 values,
    0.318 percent; ReadyMix's minimum moves from 3.1e-17 of its mean to 0.265.
25. **2026-09-11, Stage 2a. Overlap and spread are specified generation
    targets, solved for per dataset.** `[DELEGATED, 2a chose]` Average pairwise
    component overlap after Maitra and Melnykov (2010), and the coefficient of
    variation of the parent. Both replace quantities that were previously
    accidents of the location and scale ranges, and both are reported against
    what was asked for.
26. **2026-09-11, Stage 2a. `generate_dontread` is retired.** `[AUTHOR]` A
    corpus is a named, dated directory carrying its seed, configuration, git
    commit and library versions. Nothing is overwritten; the notebooks only
    read, via the tracked pointer `data/processed/CORPUS.json`.
27. **2026-09-11, Stage 2a. `mode_share_alpha` stays at 10.** `[RECOMMENDED]`
    Kept so this stage changes one thing at a time. It leaves mode dominance
    almost constant (at k = 2 the larger mode holds 0.501 to 0.760 of the
    points). Stage 2h should sweep it; alpha = 1 is the obvious other end.
28. **2026-09-11, Stage 2a. The repository size is accepted; no history
    rewrite.** `[AUTHOR]` `.git` is 391 MB, about 190 MB of it large figure and
    table blobs re-stored whole on every change. The author's decision, given
    directly: "I don't care too much about too big of a git repo." Deleting the
    15 stale figures (115 MB off the working tree) is treated as sufficient.
    A `git filter-repo` rewrite is NOT to be attempted: it would rewrite every
    commit hash and break the Zenodo deposit, for a benefit the author has said
    they do not want. Stage 4 should not revisit this unless the author raises
    it.
29. **2026-09-11, Stage 2a. The coverage figure is accepted.** `[AUTHOR]`
    `outputs/figures/CompareUQMethods_FIG_MetricCoverage.png`, reviewed and
    approved: "That output figure looks good to me." The underlying coverage
    result stands: 100 percent of the 138 empirical datasets fall inside the
    synthetic range on all nine statistical metrics, 99.3 percent on dataset
    size. The two named exceptions are ReadyMix (n = 77,439) and Elevators.
