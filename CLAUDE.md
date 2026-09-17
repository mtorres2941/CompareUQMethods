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
- **Marsh, Lewis, Hattam and Allen (in press)**, "Uncertainty characterization
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
- Be concise. Answer the question asked, at the length the answer needs. Do not
  restate what a commit message, a handoff or a table already says; point at it.
- **All communication between sessions goes through the handoff and discrepancy
  files, never through chat.** When a session ends, the author's job is to hand
  the next session a FILE, nothing else. Do not also summarize that file's
  contents back to the author as chat text: it reads as a separate set of
  instructions they have to act on, and it makes the file look incomplete. If
  something belongs in the next session's hands, it goes in the file; if it is
  already in the file, saying it again in chat is noise. The test before ending a
  stage is "could the author hand over these files and say nothing", and if the
  answer is no, the file is what needs fixing.
  When a task is done, the status line is the whole report - no closing summary,
  no re-emphasis, no "one thing worth noting". If something genuinely matters
  and is not written down, that is a defect in the document, so fix the document
  rather than narrating around it.

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

**Only the CURRENT stage's handoff is kept.** The handoffs for stages 0 through
2a-3 were deleted on 2026-09-15, by the author's decision: this repository is
published alongside the paper and a reader has no use for the editing process
that produced it. What survives a stage is the decision log in this file, which
carries every decision with its reason and its date, and the discrepancy file.
Git history retains the deleted handoffs, so nothing is unrecoverable; they are
simply not part of the deposit.

This does not relax the first rule above. An open item still may not live only
in a conversation: when a stage closes, its unresolved items move into the
DECISION LOG or into `reports/MANUSCRIPT_discrepancies.md`, not into a file
that is about to be deleted. The old "carried forward" list is replaced by that
requirement, because a chain of handoffs that no longer exists cannot carry
anything.

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

**It was suspended once, deliberately, for Stage 2a-2, and has resumed.** The
reason it was cheap: notebooks 2 and 3 had never been run against
`corpus_2026-09-12b`, so no downstream result existed for a regeneration to
invalidate. That ceased to be true the moment Stage 2a-2 closed.

**Stage 2a-3 reopened it twice more and closed it again**, because the empirical
arm changed twice inside that stage: once for a declared-unit split that the
author then rejected, and once for the rules of decision 46 that stand.
`corpus_2026-09-14d` is the result. Notebooks 2 and 3 still had not run, so again
nothing downstream was invalidated. **That was the last time.** The empirical
extract is frozen by decision 44, the category rules are settled by decision 46,
and generation is closed; neither input moves again.

| Stage | Owns | Explicitly not its job |
|---|---|---|
| **0 DONE** | Inventory, dependency map, refactor plan, baseline code and analysis assessments. Handoff deleted by decision 59 | Any change to analysis logic |
| **1 DONE** | Pinned environment, regression fixtures, persisted pLCA table, seeding machinery, correctness fixes, thin notebooks over a tested `src/`. Exactly two intended number-moving changes: neccs 1,000 to 10,000, and the wbeci assignment moved inside the loop. Handoff deleted by decision 59 | Regeneration, and every methodological judgment call. Amendment A3 settled normalization: unweighted mean, code stands, text is wrong |
| **2a DONE** | Generator audit: seeding collapse, Dirichlet concentration mismatch, stale docstring, the truncation loop, power transform, reflection, component overlap, mode counting, the 27.5 percent filter, mode-level market share, the coverage table that becomes Table 1. Then regenerate, once | Changing the fitting methods, changing the scoring target, or sweeping anything that 2h owns |
| **2a-2 DONE** | A one-off reopening of generation, by decision, because nothing downstream had been computed yet. Fresh raw empirical extract, symmetric log-space cleaning, weighted tuning objective, retune, regenerate once. Handoff deleted by decision 59 | Any fitting work, and any further regeneration. Generation closes again when this stage ends |
| **2a-3 DONE** | Resolve the EC3 categories into specifiable products, on record metadata only: drop EC3 residual bins, split concrete by specified strength, split insulation by material type. Arm 136 to 149. Regenerate as `corpus_2026-09-14d`. Two record corrections. Found that the coverage claim is false. Handoff deleted by decision 59 | Any fitting work. It is the LAST pre-2b stage: nothing after it reopens generation or the empirical extract |
| **2b DONE** | The lognormal: threshold pathology, the +0.5 offset, two-parameter versus profile-likelihood versus gamma. W1-optimal fitting alongside MLE. Also the plausibility ceiling, the support (0, inf), and the first end-to-end run of notebooks 2 and 3. Handoff deleted at the close of 2c; its findings are decisions 49 to 58 and discrepancy entries 35 to 52 | Adding new families for robustness (2h), or rescoring against a parent (2c) |
| **2c DONE** | The evaluation target: scored the synthetic arm against the known parent (recovered by replaying the generator, decision 64), cross-validated the empirical 147, the fit-versus-definitional decomposition, regret, post-stratification, overlap area, the gamma question, the bandwidth against the parent, and the scoring grid. `reports/HANDOFF_stage-2c.md` | The pLCA construction (2e) and the flip-probability threshold (2d). It did NOT split the uniform-to-variable W1 into location and shape, which is 2d's |
| **2d** | Decompose the uniform-to-variable W1 into location and shape, define the named relative measure, calibrate flip probability against relative W1, report the 1, 5 and 10 percent crossings | Building companion decision metrics (2g) |
| **2e** | pLCA construction: common random numbers across UQ methods, sweep materials per pLCA over 2 to 12, resample groupings, dominant-MUI variant, bootstrap intervals on every headline percentage and NRMSE | Changing what the headline metric is (2g) |
| **2f** | Resolve Shapiro-Wilk versus Shapiro-Francia and `_royston_pvalue`, then the multivariate model of W1 and of which method wins, to cut the metric set to three to five survivors | Regenerating, or redesigning figures (3) |
| **2g** | Sensitivity of ECI Rank #1 Frequency, magnitude-based companions, and the `(1-capecc)` divisor | Re-running the sweeps of 2h |
| **2h** | Robustness sweeps: KDE bandwidth (Scott, Silverman with a degenerate-IQR guard, cross-validated), lognormal offset, gamma and Weibull as extra families, Dirichlet concentration, multiple weight realizations, mode-to-point coupling | Anything not framed as a sweep with a tabulated result |
| **2i** (optional) | Real-building anchor, only if we decide after 2g that citing Marsh et al. (in press) is not enough | Becoming a case study |
| **3** | Figures: merge 2 and 3, rebuild 4 from the 2f survivors, the figure manifest, the naming convention, vector output, duplicate-filename check. **The figure SIZE problem is FIXED, 2026-09-15, and the diagnosis recorded here was wrong: no figure ever declared a 94 by 55 inch `figsize`. The cause was RESOLUTION. Notebook 2 set `matplotlib.rcParams['figure.dpi'] = 1200`, and `savefig.dpi` defaults to `'figure'`, so that was silently the save resolution for every figure in the notebook; notebook 3 passed `dpi=1200` to six `savefig` calls directly. All are now 300, with `figure.dpi` at 100 for the screen. Layout is measured in inches, so nothing moved but the pixel count.** | Changing any number |
| **4** (optional) | README and Zenodo re-deposit | Anything analytical. **NOT the `.git` history rewrite: declined by the author, decision 28** |

Items already known to be open and owned by a named stage, so that none of them
reads as a fresh discovery: the bandwidth rule and its KL1/KL2 inconsistency
(2h); `logfit_offset` (RESOLVED in 2b, retired; the profile-likelihood guard `PROFILE_DELTA_LO_FRAC` is what 2h sweeps in its place); the "Mode Count" label naming a
continuous modality index (2a); the 27.5 percent filter and its n cap at 749
(2a); the variance-inflation exponent and the reflection step (2a);
Shapiro-Wilk versus Shapiro-Francia and `_royston_pvalue` (2f); dependent
sampling (2e); overlap area alongside W1 (2c); the scoring grid's zero
(RESOLVED in 2b); the `(1-capecc)` divisor (2g); overlap area (RESOLVED in 2c, decision 69); W1's lack of a complexity penalty (RESOLVED in 2c, decisions 65 and 70); the linear scoring grid (RESOLVED in 2c, decision 72);
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
13. **2026-09-11, Stage 1. Support is (0, inf), open at zero. CONFIRMED BY THE
    AUTHOR 2026-09-14, in the Stage 2b prompt. NO LONGER AN OPEN ITEM.**
    - **[AUTHOR]** "I think it's (0, infinity), rather than [0, infinity) since
      we shouldn't accept zero as an input." Confirmed: "Support is settled:
      every method lives on (0, inf), open at zero."
    - **What the confirmation settled.** Stage 1 flagged this as a reach because
      it was stated in conversation rather than in a prompt, and four stages
      built on it unconfirmed. It is now an author decision in writing and stops
      being carried forward. It also fixes the inconsistency Stage 1 found, and
      says WHICH SIDE moves: the GRID changes, not the sampler. Zero is excluded
      rather than included, because an ECC of exactly zero is not admissible.
    - **What Stage 2b implemented from it.** `src/families.py`. All six methods
      are explicit truncations to (0, inf), renormalized, each exposing pdf, cdf,
      ppf and inverse-CDF sampling, so the object that is SCORED is the object
      that is SAMPLED. Sampling is by inverse CDF and not by rejection, because
      the common-random-numbers scheme Stage 2e installs needs one uniform
      variate per material per iteration mapped through every method's inverse
      CDF, which rejection sampling cannot supply. The scoring grid's lower bound
      is the first point of its own lattice above zero; see
      `fitting.score_grid_open` for why that and not an invented epsilon.
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
    coupling parameter that reduces to the old uncoupled behavior at 0.
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
    stored `dct_realeccs_trimmed.json` was already trimmed additively at the
    high end when it was written, so a log-space high bound on top would trim
    the same tail twice. Removes 342 of 107,523 values, 0.318 percent;
    ReadyMix's minimum moves from 3.1e-17 of its mean to 0.265.
    **Corrected 2026-09-12:** an earlier version said a symmetric re-clean was
    impossible because the EC3 source no longer existed. That was wrong; a
    fresh pull can be taken at any time. The recommended next step is to pull
    current EC3 data and apply the symmetric rule to raw values. The cleaning
    sensitivity says the choice of rule moves `fit_norm_SW` by about 1.5
    standard deviations. See `reports/MANUSCRIPT_discrepancies.md` entry 26.
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
    approved: "That output figure looks good to me."

    **THE COVERAGE RESULT RECORDED HERE IS NO LONGER TRUE, corrected by decision
    48.** It said 100 percent of the 138 empirical datasets fell inside the
    synthetic range on all nine statistical metrics. That was measured on the
    Stage 2a arm, whose maximum coefficient of variation was 2.40. Stage 2a-2
    rebuilt the arm from raw values and the maximum became 13.40, and nothing
    re-checked coverage until Stage 2a-3. It is now 11 uncovered dataset-metric
    pairs of 1,490. The figure must be rebuilt and the claim restated.
30. **2026-09-12, Stage 2a-2. Generation was reopened once, by decision, and is
    closed again.** `[AUTHOR]` Stage 2a's handoff says not to regenerate and
    that instruction is correct for every other stage. It was suspended here
    because notebooks 2 and 3 had never run against `corpus_2026-09-12b`, so
    nothing downstream existed to invalidate.
31. **2026-09-12, Stage 2a-2. The empirical arm is built from a frozen local
    extract of the 2026-08 EC3 store.** `[AUTHOR]` A slice of
    `../EPDsFromEC3/store`, pulled 2026-08-13/14 through the correctly-
    paginating LucidLCA wrapper: five months newer than the 2026-03 data and,
    crucially, RAW. Frozen as `data/raw/ec3_raw_ecc_2026-08-14.csv.gz`, tracked
    and checksummed, which is what makes the empirical arm reproducible from a
    clean clone.

    A fresh pull was running in the `EPDsFromEC3` repository at the same time.
    **Whether to fold it in is now settled: no. See decision 44.** The arm is
    frozen at this extract for the remainder of the project. **Do not query the
    EC3 API while a pull is running there:** EC3 rate limits per account, not
    per process. Access itself is open; see decision 45.
32. **2026-09-12, Stage 2a-2. The empirical arm is valid-at-pull-date, matching
    the 2026-03 scope.** `[AUTHOR]` The 2026-08 pulls include expired
    declarations and the 2026-03 pull did not, so keeping them would have
    changed the population and the cleaning rule at once. 38.6 percent of the
    slice is expired and is dropped. The with-expired variant is a reported
    sensitivity, not the primary.
33. **2026-09-12, Stage 2a-2. Cleaning is a symmetric log-space 3 x IQR bound.**
    `[AUTHOR]` This is what decision 12 asked for and what Stage 2a could only
    half-apply. On raw values it removes 816 of 120,280, 544 low and 272 high.
    **The empirical arm is 136 categories, not 138**: `Siding` and
    `SinglePlyOther` fall below three values, both through expiry.
34. **2026-09-12, Stage 2a-2. EPD-level uniform weighting stays the primary
    definition.** `[AUTHOR]` 55 percent of records share a (manufacturer, GWP)
    pair, so the uniform-weighted baseline is already implicitly weighted by
    publication frequency. It is nonetheless what a practitioner pulling from
    EC3 actually holds, which makes it the right baseline for a paper about
    what practitioners should do. The implicit weighting is stated in the text;
    the deduplicated variant is a Stage 2h sensitivity.
35. **2026-09-12, Stage 2a-2. The tuning objective weights the characteristics.**
    `[AUTHOR]` `crit_bw_1` and `coeffvar` at 3, `modality_index` at 2, `n` at
    0.25, the mode-count total variation as its own term at 3, everything else
    at 1. Modality and spread drive the KDE-versus-parametric comparison
    directly; the goodness-of-fit characteristics are largely downstream of
    them. Every run reports the objective weighted and unweighted.
36. **2026-09-12, Stage 2a-2. The overlap range comes down to [1e-2.5, 1.4],
    reversing Stage 2a's decision 25 on this parameter.** `[DELEGATED, 2a-2
    chose]` Stated explicitly because CLAUDE.md forbids reversing an earlier
    decision silently. Stage 2a raised the overlap to [0.3, 1.4] to match an
    empirical arm measured as 81.9 percent unimodal; that figure was an
    artifact of the additive high-end trim in the stored file. On raw data
    cleaned symmetrically the arm is 49.3 percent unimodal, and the directly
    fitted empirical overlap has a 95th percentile of 0.2671, so the Stage 2a
    range sat entirely above the empirical body. Two independent measurements
    agree, and both were wrong before for the same reason.
37. **2026-09-12, Stage 2a-2. The lognormality cost of matching modality is
    accepted and recorded.** `[DELEGATED, 2a-2 chose]` `fit_lognorm_SW` worsens
    from 0.933 to 1.828 standardized W1 and is now the worst characteristic.
    Real ECC datasets are 49 percent multimodal while keeping a median
    Shapiro-lognormal statistic of 0.937, so their modes are gentle shoulders
    on a lognormal body; the generator reaches the same mode COUNT by
    separating components, which is a different shape. Four attempts to recover
    it all made it worse, so the cost is structural rather than a tuning
    artifact. Related, same cause: 6.5 percent of the corpus has six or more
    modes against an empirical 0.7 percent. **Owner: 2h.**
38. **2026-09-12, Stage 2a-2. Modality is steered by VISIBLE modes, not by
    Silverman's test.** `[AUTHOR]` `modality.n_modes_visible` counts local
    maxima of a Scott's-bandwidth KDE with a prominence threshold of 5 percent
    of the peak. It is the author's original `estimate_maxima` with the
    continuous index replaced by a count, and Stage 2a's decision 23 to drop
    that metric entirely was too broad: the defect was the readout, not the
    idea. Silverman's critical bandwidth stays as a reported characteristic.

    The cost of getting this wrong was most of Stage 2a-2. Both measures agree
    that about half the empirical datasets are multimodal by Silverman, but
    94.9 percent of them have exactly ONE visible mode, because their structure
    is shoulders on a right-skewed body rather than separated humps. Tuning
    against Silverman drove the component overlap down through five
    configurations, each of which looked like an improvement on the statistics
    being watched and made the corpus visibly worse. The author reported the
    shapes were wrong three times before the right measure was computed.
39. **2026-09-12, Stage 2a-2. Stage 2a's overlap range [0.3, 1.4] is restored**,
    reversing decisions 36 and its successors within this stage. `[AUTHOR]`
    Visible-mode total variation against the empirical arm: 0.280 at
    [1e-2.5, 0.9], 0.022 at [0.3, 1.4]. Every characteristic improved, and
    `w_v_uw_wasserstein`, the paper's central quantity, went from 0.550 to
    0.131 standardized W1. Stage 2a had this right.
40. **2026-09-12, Stage 2a-2. Components must have a bounded density.**
    `[AUTHOR]` beta with a < 1 or b < 1 and beta-prime with a < 1 are J-shaped:
    ordinary moments, infinite density at an endpoint, drawn as spikes. 11.3
    percent of components were shaped that way. Refused and redrawn.
41. **2026-09-12, Stage 2a-2. Draft corpora at 1,000 datasets for iteration.**
    `[AUTHOR]` `python corpus.py <label> 1000`, 110 s against 850 s. The label
    records it. A draft must never be used for a paper number.
42. **2026-09-13, Stage 2a-2. Both arms are trimmed by the same multiplicative
    log-space rule.** `[AUTHOR]` `trunc_rule = 'log'` with
    `min_q1_over_iqr = 0.5`. The author raised this three times before it was
    rechecked, and was right each time: trimming the two arms by different rules
    put a difference between them on a dimension the study is about.

    The claim that a multiplicative rule could not work was false. It rested on
    an assertion that q3/q1 tends to 1 under the shift so the bounds collapse
    onto the interquartile range; in fact the multiplicative bound converges to
    the additive one from above and never collapses. What actually failed was
    `MixtureParent.truncated_moments`, which integrated on a uniform grid and
    returned sd = 0 whenever the truncation bounds were wide relative to the
    body. It now integrates on the components' own quantiles. That bug was
    latent under the additive rule and would have bitten any Stage 2h sweep of
    `trunc_iqr_mult`.

    Effect: mean W1 across the ten characteristics fell from 0.488 to 0.270,
    with every characteristic improving. Several earlier rounds of generator
    tuning in this stage were compensating for a difference created by the
    inconsistent cleaning.


43. **2026-09-13, Stage 2a-3. A category that is not one product population is
    resolved into the products it holds, on record metadata only.**
    `[AUTHOR]` **SUPERSEDED IN ITS AXIS BY DECISION 46**, which is what is
    implemented; the constraint below stands and is the reason the work is
    defensible at all.

    **A split may read only metadata carried on the EPD record or on EC3's
    category tree, never the ECC values.** This study measures the modality,
    dispersion and skewness of ECC distributions, so splitting a category
    because its values look bimodal and then reporting that ECC datasets are
    unimodal is circular.
44. **2026-09-13, Stage 2a-3. The empirical arm is FROZEN at the 2026-08
    extract for the remainder of the project.** `[AUTHOR]` This closes the
    open item in decision 31 and in the Stage 2a-2 handoff, which left folding
    in a newer EC3 pull undecided. The answer is no. The extract is archived,
    checksummed and reproducible from a clean clone, and one further month of
    declarations will not move the characteristic distributions enough to
    justify re-invalidating the calibration. If the arm is refreshed at all it
    will be once, deliberately, before submission, as an author decision.
45. **2026-09-13, Stage 2a-3. Direct EC3 API access is NOT closed to this
    account.** `[AUTHOR]` Decision 31 as originally written said it was, on one
    403 response to one request with a stale key. A working key has since been
    set. Corrected in the decision, in `CONTEXT.md`, in
    `data/raw/ec3_raw_ecc_2026-08-14_runmeta.json` and in
    `reports/MANUSCRIPT_discrepancies.md` entry 30. It changes nothing about
    decision 44: the arm is frozen by choice, not by access.

46. **2026-09-14, Stage 2a-3. The material categories are resolved into
    SPECIFIABLE PRODUCTS, by three metadata rules. The arm is 149 datasets.**
    `[AUTHOR]` This replaces the declared-unit axis of decision 43, which the
    author rejected: "Separating by declared unit doesn't quite seem reasonable.
    Why is it strange that some aggregates might be declared per 1 kg and some
    per 1000 kg? That still might be the same material."

    **The criterion is not dispersion.** The author's words: "This exercise isn't
    about fixing dispersion at all, it's separating material categories
    meaningfully." Splitting `ReadyMix` by compressive strength moves its
    coefficient of variation only from 0.29 to 0.27 and is still right, because
    4000 psi and 5000 psi concrete are different products and strength is the
    primary characteristic a structural engineer specifies concrete by. A
    dataset stands for one material choice in a pLCA, so the test is whether a
    category is something a specifier could name. **A later stage must not
    reintroduce a dispersion screen.**

    1. **EC3 residual bins are dropped.** A non-leaf node of EC3's category tree
       holds the EPDs EC3 did not place in any child, so it is a residual bin
       rather than a product. 15 dropped where the children are in the arm,
       including `Insulation` (666 records, CV 7.65) and `Steel` (576, 1.57).
       This is what resolves insulation and steel, and the steel children are
       exactly the distinctions a structural engineer draws. Four parents with
       no child in the arm are KEPT, because dropping them would remove the
       material; their heterogeneity is a stated limitation.
    2. **Concrete is split by specified 28-day compressive strength**, a
       structured EC3 field at 90 percent or more. `ReadyMix`, `Shotcrete`,
       `ConcretePaving`, `CMU`. `CementGrout`, `FlowableFill` and `OilPatch`
       carry the field but are not specified this way in building design and are
       left whole.
    3. **Insulation is split by material type** from the product name and
       description, with a fixed declared pattern list. Records naming no
       material become an explicit "type not stated" dataset rather than being
       dropped, because that is 131 of 335 `BoardInsulation` records.

    Thickness was tested as a fourth rule and rejected: it explains the
    `Insulation` bin's spread, but that bin is dropped by rule 1 and thickness
    parses for only 96 of 335 board records.
47. **2026-09-14, Stage 2a-3. Generation reopened twice more and is now closed.**
    `[AUTHOR]` Three corpora in this stage, against a standing rule of one,
    because the empirical arm changed twice after the first. `corpus_2026-09-14d`
    is active: 9,999 datasets, not 10,000, because one parent failed to solve and
    was reported rather than approximated. Notebooks 2 and 3 had still never run,
    so nothing downstream was invalidated. **The retune is worth -0.0038 against
    the resolved arm, marginally WORSE and inside the 0.0066 seed noise; it is
    kept because the alternative corpus's parameters cite an arm that was
    withdrawn.** See `MANUSCRIPT_discrepancies.md` entry 34.

48. **2026-09-14, Stage 2a-3. The corpus is accepted as matching the empirical
    arm well enough, and the coverage shortfall is stated in the text rather
    than engineered away.** `[AUTHOR]` "If you think the synthetic datasets
    match the empirical datasets well enough, let's just move on."

    The judgment behind it: mean standardized W1 across the ten characteristics
    is 0.2326 and the VISIBLE mode distribution, which the generator is steered
    by, matches to a total variation of 0.0128. The one real gap is the upper
    tail of dispersion, and `audits/dispersion_reach.py` shows it is
    not reachable by any parameter: eight candidates move the achieved sample
    coefficient of variation from 1.65 to at most 2.15 against an empirical
    14.34, and none puts a single dataset above 3. Closing it would be a
    generator REDESIGN, heavier-tailed parents or a different truncation rule.

    It is also the right thing to state rather than chase, because the five
    uncovered datasets are `Aggregates`, `PowerCabling`, `Grouting`, `Elevators`
    and `Chairs`: exactly the categories decision 46 could not resolve into
    products. What the corpus cannot reach is the shape of a contaminated EC3
    category, not the shape of a material.

    **This is option A of the three tabled in `MANUSCRIPT_discrepancies.md`
    entry 34.** Options B, generator redesign, and C, excluding the uncovered
    categories, are declined. The manuscript owes a restated coverage claim and
    a rebuilt Figure `CompareUQMethods_FIG_MetricCoverage.png`; decision 29 is
    corrected in place. **Generation is closed and no further retuning is
    warranted: four retunes in Stage 2a-3 all moved the objective by less than
    one seed-to-seed standard deviation and two made it worse.**

49. **2026-09-14, Stage 2b. Physically implausible records are removed by an
    EXTERNAL bound, on mass-declared categories only.** `[AUTHOR]`
    `empirical.MASS_ECC_CEILING = 100.0` kgCO2e/kg, applied before cleaning.

    **The bound may not be read off the data.** This study measures dispersion
    and modality, so a ceiling taken from the arm's own quantiles, standard
    deviations or visible gaps would be circular in the same way a
    dispersion-based split would have been (decision 46). It is anchored on
    published embodied-carbon inventories, where the highest building-product
    coefficients are of order 13 kgCO2e/kg for primary aluminium (ICE v3.0), and
    cross-checked stoichiometrically: 100 kgCO2e per kg of delivered product
    needs about 27 kg of pure carbon burned per kilogram shipped. Set at 100
    rather than 25 so it cannot be read as a tuned threshold. **The ICE figure
    is from the analyst's knowledge and `refs/` holds no copy; verify it against
    the source before it goes in the paper.**

    Applied ONLY where an external bound exists. For volume, area, length and
    item declarations there is none, so none is invented: the ten highest and
    ten lowest records per unit type are REPORTED rather than filtered, by
    `audits/plausibility_ceiling.py`, and left in the arm.

    **Corrected 2026-09-15: that report was computed on the RAW extract and so
    listed records that never reach the arm.** Two filters already stand between
    a raw record and a dataset -- this ceiling, and the symmetric log-space
    3 x IQR rule of decision 33 -- so a raw extreme is not an open question. The
    script now applies both before taking extremes and reproduces the production
    arm exactly, 149 datasets and 117,079 values. The extremes that DO survive
    are `Aggregates`, `PowerCabling`, `Grouting` and `Chairs`, which is decision
    48's stated limitation and not a new decision: what is extreme there is the
    shape of a contaminated EC3 category, not the shape of a material.

    **Both gates were checked.** The catch is 115 of 117,807 raw records, which
    is 11 of 117,090 cleaned values, 0.0094 percent against a 0.1 percent
    stop-and-report threshold: PASSED. The calibration gate did NOT pass, and
    nothing was done about it, which is the instruction: the tuning objective
    moved 0.2075 to 0.2147, or 1.10 of the 0.0066 generator seed-to-seed
    standard deviation. Generation stays closed (decisions 47 and 48) and
    reopening it is an author decision. See `MANUSCRIPT_discrepancies.md`
    entry 35 for why 1.10 sd overstates it.
50. **2026-09-14, Stage 2b. Decision 13 is CONFIRMED and IMPLEMENTED: the
    support is (0, inf), open at zero.** `[AUTHOR]` Every one of the six methods
    is now an explicit truncation of its parent to (0, inf), renormalized,
    exposing pdf, cdf, ppf and inverse-CDF sampling. `src/families.py`.

    Sampling is by inverse CDF and never by rejection. The two give the same
    distribution; what rejection cannot do is take ONE uniform variate per
    material per iteration, which is what the common-random-numbers scheme of
    Stage 2e needs. `rvs_from_uniform` is that entry point.

    **This moved almost nothing, and that is itself the finding.** The Stage 1
    grid started at exactly zero, so the normal and the KDE were ALREADY being
    scored truncated and renormalized; making it explicit changes the scored
    values by 0.000, and opening the grid at zero moves them by 0.1 to 0.4
    percent of the median. What was wrong was the description, not the
    arithmetic: the manuscript describes an untruncated normal. Discrepancy
    entry 18 is resolved and the grid, not the sampler, is what changed.
51. **2026-09-14, Stage 2b. The lognormal is the three-parameter fit with the
    threshold chosen by PROFILE LIKELIHOOD, and `LOGFIT_OFFSET` is retired.**
    `[AUTHOR]` The likelihood of a three-parameter lognormal is unbounded as the
    threshold approaches the smallest observation, so the global MLE does not
    exist; the author named the treatment wanted and it is implemented in
    `families.fit_lognorm3_profile`.

    Two measured results that a later stage must not reverse by forgetting:
    **the two-parameter lognormal is NOT the simple answer** -- it is the worst
    of the three lognormals and worse than gamma -- and **the offset was
    patching near-zero values, not the threshold pathology**, which the Stage 1
    code could not have exhibited because it never estimated a threshold.
    Entries 37 and 40.

    **The guard is not cosmetic, and setting it too close to `min(x)` produced
    a model that scored well and was unusable.** At 0.01 of a standard deviation
    the fits with no interior maximum reached a model standard deviation of
    3,281 on data whose own is 0.6, because W1 barely charges for a thin far
    tail while the pLCA samples from it. Found in the pLCA results, not in the
    fit scores. `PROFILE_DELTA_LO_FRAC = 0.25`, the smallest guard at which no
    fitted model on either arm exceeds five times the data's standard deviation;
    it also improves mean W1 on both arms. Chosen on the bounded-variance
    criterion and NOT on W1, so that it is not tuned to the score it is judged
    by. Discrepancy entry 43.

    At 0.25 the guard determines the threshold for **48 percent of empirical
    fits**, so for about half the arm the likelihood does not identify a
    threshold and it is set a fixed fraction of a standard deviation below the
    smallest observation. **The paper must describe that as a scale-aware
    version of the heuristic the offset was, not as an estimate.** Reported per
    fit in `params[...]['status']`. **Stage 2h sweeps
    `PROFILE_DELTA_LO_FRAC` where it would have swept the offset.**
52. **2026-09-14, Stage 2b. Maximum likelihood stays the study's estimator, and
    W1-optimal fitting is reported ALONGSIDE it.** `[RECOMMENDED]`
    `fitting.FIT_METHOD = 'mle'`. Maximum likelihood is what a practitioner
    would do; direct W1 minimization is the fair-comparison control that answers
    the objection that the parametric families were judged by a rule they were
    never fitted under. Both are implemented and both are reported.

    **It matters: on the empirical arm every parametric family beats the KDE
    once fitted by W1, and three of five beat it even under maximum
    likelihood.** Entry 42. **2c owes the out-of-sample version** before any of
    it goes in the paper, because W1 is an in-sample criterion with no
    complexity penalty and the families differ in flexibility.
53. **2026-09-14, Stage 2b. The pLCA remainder is held out, named, and printed.**
    `[DELEGATED, 2b chose]` The corpus holds 9,999 datasets, so it does not
    divide by four. Every group stays at exactly four and `dataset813`,
    `dataset2876` and `dataset7985` are in no pLCA. A short last group would put
    a rank-1 frequency out of three in the same column as one out of four, and
    every headline metric is a frequency over ranks among exactly four
    materials. 2,499 pLCAs, not 2,500.

54. **2026-09-14, Stage 2b. The KDE bandwidth is Silverman's robust rule, guarded
    by a minimum EFFECTIVE sample size.** `[AUTHOR]` `BW_METHOD =
    'silverman_guarded'`, `customstats.SILVERMAN_MIN_NEFF = 30`. **The threshold
    is 20 from 2026-09-16; decision 80 supersedes this one on that value only.**
    This resolves
    the KL1 / KL2 / this-paper inconsistency of decision 9 and entry 10 in favor
    of the rule Torres et al. (2026) uses and defends.

    **The guard swaps the SCALE ESTIMATE, not the rule.** It is Silverman's
    `0.9 * scale * n_eff ** -0.2` throughout; the only thing the threshold
    decides is whether `scale` is the robust `min(sd, IQR/1.34)` or the plain
    standard deviation. Without the `min()` the two rules differ only in their
    coefficient, 0.9 against 1.06, so this keeps one rule with one conditional
    inside it rather than switching between two rules at a threshold. Falling
    back to Scott instead scores marginally better on held-out likelihood
    (-0.720 against -0.758 in the mean) and worse on W1 (0.186 against 0.173);
    the 0.9 form is used because it is better on the criterion the study
    reports and because one rule is what a reader can check.

    **The guard is on sample size, not on the interquartile range, and that is
    the opposite of where the problem looks.** Where `(IQR/1.34)/sd` is smallest
    -- heavy-tailed categories with a tight core, `PowerCabling` at 0.008 with
    n = 400 -- Silverman beats Scott on held-out likelihood 100 percent of the
    time, and flooring the robust scale makes things WORSE. It fails at n = 3 to
    10, where the quartiles are interpolated between two order statistics. The
    guard uses the KISH EFFECTIVE sample size, because a concentrated Dirichlet
    draw can leave three effective observations in a 200-value dataset.

    **The threshold is calibrated on leave-one-out likelihood, NOT on W1**, and
    that distinction has to survive into the manuscript: W1 falls monotonically
    as the bandwidth shrinks, so it cannot choose a bandwidth and would have
    picked a rule that produces spikes. Discrepancy entries 44 and 45.

    **Numbers.** Empirical `KDE, Variable` mean W1 0.2507 to 0.1530, median
    0.1487 to 0.0866, mean rank 2.738 to 1.805; synthetic 0.1017 to 0.0778,
    0.0635 to 0.0374, rank 2.123 to 1.445. Only the two KDE columns move;
    Lognormal and Normal are bit-identical.

    **The manuscript owes an explanation of the minimum sample size.** It is a
    stated methodological choice with a number in it, and a reviewer will ask.

55. **2026-09-15. The corpus is regenerated as `corpus_2026-09-15` and holds
    10,000 datasets.** `[AUTHOR]` "I'm fine regenerating the corpus. It
    shouldn't move the downstream numbers too much if we've done our job right."
    It did not.

    **The cause was a rejected draw treated as a failure.** One parent draw
    failed with `component_targets_exhausted` -- the skewness and kurtosis drawn
    for a component could only be met by a J-shaped density, which is refused --
    and `generate_dataset` retried only when the status was `mode_too_narrow`.
    The component targets are themselves random, so a fresh draw is the right
    response. `generator.REDRAWABLE` now names both statuses, and
    `corpus.generate_corpus` records the REASON for any slot it does abandon,
    which it previously counted without explaining.

    Refusing to APPROXIMATE a target that cannot be met is a different principle
    and is untouched.

    **What moved.** The random stream diverges from the redraw onward, so about
    half the datasets are different draws. Mean W1 by method changes in the
    fourth decimal, mean rank by at most 0.007, median coefficient of variation
    0.5018 to 0.5032. That stability IS the result: it says the generator is
    stationary and no single dataset was carrying a conclusion. The empirical
    arm is bit-identical.

    **The pLCA is 2,500 groups covering all 10,000 datasets**, so the three
    held-out datasets are gone and the manuscript's "2,500 pLCAs" is correct
    again.
56. **2026-09-15. `outputs/` is written by the notebooks and by nothing else.**
    `[AUTHOR]` "Everything should be traceable back to the notebooks. The
    notebooks should reproduce the entirety of this analysis." Two figures dated
    2026-03 had no producer anywhere in the repository and were deleted; one
    figure had been written by a scratch script and its code is now a notebook
    cell. **Audit scripts may write only under `outputs/tables/audits/`, never
    to `outputs/figures/` or the top level of `outputs/tables/`.**
57. **2026-09-15. `weight_outliers` was measured from the ECDF's padding, and is
    corrected on BOTH arms.** `[DELEGATED, chose to fix]` `weighted_ecdf` pads
    its arrays with `-inf` and `+inf` so the interpolator it returns extrapolates
    flat outside the data. `empirical_metadata` interpolated its quartiles on
    those padded arrays, so whenever the smallest value carried more than a
    quarter of the weight, `q1` came back as `-inf`, the interquartile range
    became infinite, both outlier comparisons were silently False, and the
    metric was reported as ZERO. Where both quartiles landed in the padding the
    subtraction produced NaN, which is the warning that exposed it.

    **It is a measurement fix, not a methodological change, and it is applied
    identically to both arms.** Confined to small datasets: the largest affected
    dataset has n = 39 and most have n = 3.

        synthetic  weight_outliers      474/10000   0.0540 -> 0.0609
        synthetic  weight_outliers_uw   163/10000   0.0526 -> 0.0580
        empirical  weight_outliers        3/  149   0.0614 -> 0.0656
        empirical  weight_outliers_uw     1/  149   0.0590 -> 0.0612

    Both arms move the same way by a similar amount, so the arm-to-arm agreement
    the generator was tuned against is essentially unchanged. **Generation stays
    closed and nothing is retuned**, per decisions 47, 48 and 55. No W1 column
    moves on either arm. The manuscript owes nothing here except the corrected
    numbers; see `reports/MANUSCRIPT_discrepancies.md`.

58. **2026-09-15. A corpus's characteristics can be RECOMPUTED without redrawing
    it, and `corpus_2026-09-15b` is that and nothing more.** `[DELEGATED, chose]`
    Read this before concluding that generation was reopened. **It was not.**

    The empirical arm computes its characteristics when notebook 1 runs; the
    synthetic arm stores them in `metrics.parquet` at generation time. So a
    correction to the metric code reaches one arm and not the other, and the two
    would be measured by DIFFERENT CODE on a dimension the study reports, which
    is the failure decision 42 is about.

    `corpus.remetric_corpus` reads the values already on disk, reruns the current
    `empirical_metadata` over them, and copies the generation record forward. No
    dataset is redrawn and no random number is consumed. A new directory is
    written rather than the source edited, so the old readout stays on disk
    beside the new one and the change is diffable.

    **Verified, and the verification is the point:** `values.parquet`,
    `parents.json.gz`, `combos.csv` and `invalid_datasets.json` are all BYTE
    IDENTICAL between the two directories, the only columns that differ are the
    two in decision 57, and notebook 3 then reproduced `TABLE_PLCAResults.csv`
    byte for byte. A regeneration could not do any of that.

    A later stage that needs a metric corrected uses this, not `generate_corpus`.

59. **2026-09-15. The closed stages' handoffs and the audit TABLES are deleted,
    and `audits/` is flat.** `[AUTHOR]` "People reading this repository don't
    need to know anything about our editing process", and, on the audit tables,
    "I don't know what any of these are for."

    Deleted: `HANDOFF_stage-0` through `HANDOFF_stage-2a3`, `REVIEW_stage-2a.md`,
    and the four `outputs/tables/stage2*` directories, 98 tables. `audits/` went
    from 43 scripts in four stage-named directories to 18 in one flat directory,
    each named for what it measures.

    **The rule for keeping an audit script was whether something PERMANENT cites
    its numbers** -- this decision log, `CONTEXT.md`, `src/`, or
    `MANUSCRIPT_discrepancies.md` -- because that is what makes a decision
    reproducible rather than asserted. A 12 KB script that rewrites its table on
    demand is a better record than a 4 MB CSV, which is why the tables could go
    and the scripts could not.

    **What this does NOT relax:** nothing outstanding may live only in a
    conversation. An open item now lands in this decision log or in the
    discrepancy file. See the amended "Continuity across sessions" section.

    Kept deliberately: `reports/baselines/TIMING_baseline.md`, because
    the author wants a before-and-after comparison of the repository.
60. **2026-09-16. A record may be judged on its VALUE against an EXTERNAL bound.
    Decisions 43 and 46 were being read far too broadly, and that reading is
    withdrawn.** `[AUTHOR]` "Why would we not judge a record on its value? If a
    record says a concrete mix is 162,236 kgCO2e/m3, that value is sufficient to
    discard that record."

    **Decisions 43 and 46 govern SUBCATEGORISATION, not filtering.** They are
    about how a category is resolved into products -- insulation by material
    type, concrete by strength -- and they forbid choosing those axes by looking
    at the spread of the values. An earlier session cited them as a blanket
    prohibition on judging any record by its value. They say nothing of the
    kind, and decision 49's mass ceiling has been judging records by their value
    since it was written.

    **The line that actually matters is where the THRESHOLD comes from.** An
    absolute bound anchored outside the data is legitimate: 100 kgCO2e/kg comes
    from published inventories and a stoichiometric check. A bound read off the
    arm's own quantiles, spread, or distance from a median is circular, because
    dispersion is what the study measures.

    **The author's sequencing is the correct one and is what the code does:**
    resolve categories into products FIRST, then filter implausible values.

    **There are two filters and they have different jobs.** Decision 33's
    symmetric log-space 3 x IQR is the general-purpose outlier rule and it works:
    on `ReadyMix [4000-4999 psi]` its upper bound sits at 3x the median and it
    trims 42 records. **It fails, structurally, on a contaminated category**,
    because a relative filter's width is set by the spread of the very
    contamination it is meant to remove: on `Aggregates` the same rule puts its
    upper bound at 41,238,610x the median and trims nothing. That is the
    mechanism behind every extreme this project has chased, and it is why the
    fix is a category rule rather than a tighter filter.

61. **2026-09-16. Two categories are dropped as not one product population, and
    a named-product exclusion list is applied to a third. The arm is 147
    datasets.** `[AUTHOR]` "Please drop those two and apply the exclusion list to
    the other three."

    `categorysplit.NOT_ONE_POPULATION` drops `Chairs`, where only 15 of 86
    records name any kind of seating and the rest are kitchen mixer taps,
    asphalt, culverts, hollowcore slabs and bathroom furniture; and `Grouting`,
    where only 44 of 225 name a grout and the rest are gypsum plasters,
    decorative renders, GGBS, concrete admixtures, epoxy coatings, a cable clamp
    and a glazed door. These are EC3 LEAF categories, not residual bins, so no
    rule on the category tree could have caught them; the evidence is the
    product name, which decision 43 permits.

    `categorysplit.EXCLUDED_PRODUCTS` removes 7 records from `Aggregates`:
    sinks, washbasins and porcelain stoneware slabs, which are finished products
    rather than crushed stone, gravel and sand.

    **It is an EXCLUSION list and not an inclusion vocabulary, and that is not a
    stylistic choice.** An inclusion rule must anticipate every legitimate
    naming convention in every language. Tried first, it removed 22
    `PowerCabling` records that are plainly cables -- `Cable a Haute Tension`,
    `TSLF 24kV`, `NF C 33-226`, `Nexans U-1000 R2V`, `H07RN-F` -- and two
    Schindler elevators whose names are model numbers, while MISSING the real
    intruders, because `GRANITEK Sinks` and `Composite granite kitchen sinks`
    both match on "granite". An exclusion list removes only what it names.

    **`PowerCabling` and `Elevators` were checked and need no entry.** All 381
    distinct cable names are cables or conductors; all 20 elevator names are
    elevators. Applying a list to a clean category can only do harm.

    **The rule provably cannot see a value.** `tests/test_categorysplit.py`
    drives the whole assignment on a frame with no `ecc` column at all.

    **Numbers.** 149 datasets to 147; 117,079 values to 116,768. The generator
    calibration IMPROVES and stays far inside the gate: weighted objective
    0.2308 to 0.2278, a move of 0.45 of the 0.0066 seed-to-seed standard
    deviation. **No regeneration**, consistent with decisions 47, 48 and 55.

62. **2026-09-16. The canonical length unit is the METRE, not the inch.**
    `[AUTHOR]` "If you want to change length2in to length2m and change values to
    meters or something like that, I'm totally fine with that."
    `funcs_unit_conversion.length2m`. Nothing else in the study is imperial and
    a cable emission factor per inch is not a quantity anyone checks by eye.

    The frozen extract stores `ecc` already computed, and it CANNOT be rebuilt:
    the EC3 store it came from has moved on, so its sha256 no longer matches and
    a rebuild would change the POPULATION rather than just the unit, which
    decision 44 refuses. `empirical.fix_length_unit` therefore rescales the
    1,500 length-declared rows on the way in.

    **This moves no analysis number.** Every dataset is normalized by its own
    unweighted mean and cleaned by a log-space rule, and both are invariant
    under a constant rescale of a whole dataset; the full fixture regression
    passes unchanged. `PowerCabling` now reads a median of 2.36 kgCO2e/m instead
    of 0.0599 kgCO2e/in. `psi` and `rval` are deliberately kept: concrete is
    specified in psi, which is what decision 46's strength classes are named in,
    and R-value is the convention for thermal resistance.
63. **2026-09-16. A record whose DECLARED UNIT contradicts its own published
    mass is removed. Two records, and they were holding up the arm's dispersion
    tail.** `[AUTHOR]` "14,300 kgCO2e/m is completely insane and obviously needs
    to be excluded."

    EC3 publishes a second, independent number on each EPD: the GWP per
    KILOGRAM. `THHN/THWN-2 High Speed (HS)` declares 14,300 kgCO2e for one metre
    of ordinary building wire while publishing 4.017 kgCO2e/kg, which puts 3,560
    kg of copper in a single metre. The material is fine; the DECLARED UNIT is
    wrong, and the record's own two numbers say so without reference to any
    other record.

    **BOTH published figures must be impossible before a record is removed, and
    testing proved that necessary.** The implied-mass signal alone flagged four
    `ReadyMix` records at a wholly normal 372 to 451 kgCO2e/m3, because their
    `gwp_per_kg` field reads 0.020 where concrete is about 0.105: dividing by a
    broken field manufactures an absurd mass from sound data. The ECC signal
    alone is a per-unit ceiling that cannot be anchored externally, since a
    square metre of product has no general size. `empirical.MAX_MASS_PER_UNIT`
    and `MAX_ECC_PER_UNIT`.

    **It is not a dispersion screen and that is tested, not asserted.** Scoring
    one record alone and scoring it inside a crowd of fifty near-identical rows
    gives the same answer, which no filter keyed on a median or an interquartile
    range could do. `tests/test_categorysplit.py`.

    **The live EC3 store is NOT a dependency.** It has already moved on from the
    frozen extract, and reading it at analysis time would break reproducibility
    from a clean clone, which is the point of decision 31. The values are frozen
    as `data/raw/ec3_gwp_per_kg_2026-09-16.csv.gz`, tracked and checksummed,
    covering 96.5 percent of the extract. Records without a usable value are
    simply not checked.

    **WHAT THIS MOVED, AND IT IS MUCH MORE THAN TWO RECORDS.** `PowerCabling`
    n 400 to 398, coefficient of variation **13.404 to 2.030**, skewness 18.33 to
    4.34, kurtosis 347.0 to 28.3. The ARM'S MAXIMUM coefficient of variation,
    which is the number decision 48 rests on, falls from **13.404 to 6.929**, and
    the maximum is now `Aggregates` rather than `PowerCabling`. Coverage improves
    from 11 uncovered dataset-metric pairs of 1,490 to **5 of 1,470**, with only
    one dataset now uncovered on dispersion. Values 116,768 to 116,766.

    **DECISION 48'S PREMISE IS PARTLY WITHDRAWN.** It recorded that no generator
    parameter could reach an empirical coefficient of variation of 14.34. That
    figure was inflated by two mislabeled records. The real target is 6.93, the
    gap is about half what it was, and the uncovered categories are no longer
    what decision 48 listed. **The coverage claim and the figure must be restated
    from the rebuilt tables, not from decision 48's numbers.**

    **THE CALIBRATION APPEARS TO MOVE OUTSIDE THE GATE AND THAT IS AN ARTIFACT.
    DO NOT RETUNE ON IT.** The weighted objective goes 0.2278 to 0.2425, a
    nominal worsening of 2.2 of the 0.0066 seed-to-seed standard deviation, and
    an earlier version of this entry reported that as a real case for
    regenerating. It is not.

    `coverage.distribution_comparison` divides the Wasserstein distance by the
    EMPIRICAL STANDARD DEVIATION, so the standardized number moves when the
    denominator moves. Removing a coefficient of variation of 13.4 from a
    147-value distribution shrinks that standard deviation from 1.2554 to
    0.7067. Measured in ABSOLUTE terms the corpus now matches the corrected arm
    BETTER on exactly the characteristic driving the change:

        coeffvar    absolute W1  0.3463 -> 0.2690   IMPROVED by 22 percent
                    empirical sd 1.2554 -> 0.7067
                    standardized 0.2759 -> 0.3806   worse only by division

    `coeffvar` and `crit_bw_1` carry weight 3 in the objective (decision 35),
    which is why the two of them set the headline. `crit_bw_1` and `entropy`
    move slightly the wrong way in absolute terms, 0.0744 to 0.0872 and 0.3452
    to 0.3606, and those are inside the noise.

    **So generation stays closed and no retune is warranted**, decisions 47, 48
    and 55 unchanged. This is the failure mode decision 38 already records from
    Stage 2a-2: a configuration that looks like an improvement on the statistic
    being watched while the corpus gets no better. **A later stage that sees the
    0.2425 and reaches for the tuner should read this paragraph first.**

64. **2026-09-16, Stage 2c. The parent of a synthetic dataset is RECOVERED by
    replaying the generator, because the corpus does not store enough to rebuild
    it.** `[DELEGATED, 2c chose]` Read this before concluding that generation was
    reopened. **It was not**, and the same reasoning as decision 58 applies.

    `parents.json.gz` stores how a parent was ASKED for: each component's moment
    targets, from which `components.solve_component` recovers its location and
    scale deterministically, plus the global shift and the truncation bounds. It
    does NOT store the displacement the overlap solve gave each component, which
    is one solved scalar times k ordinates drawn from the generator's stream, and
    one recorded overlap value cannot identify k - 1 displacements.
    **CONTEXT.md's claim that the record was "enough to rebuild its CDF exactly"
    was false.**

    `corpus.rebuild_parents` replays the generation loop, which is deterministic
    given the seed, and keeps the parent objects `generate_corpus` discarded. No
    corpus is written, no dataset is redrawn, and no random number reaches a
    result. **The replay is CHECKED, not trusted:** it refuses to run unless
    `genconfig.DEFAULT` still equals the configuration recorded in the corpus, it
    compares twelve record fields plus `pi`, `market` and `mode_counts` per
    dataset, and it compares the replayed values and weights against
    `values.parquet` element by element. All 10,050 datasets of
    `corpus_2026-09-15b` replay byte-identically, in 13 minutes. Cached as
    `parents_spec.json.gz` inside the corpus directory, which adds a derived file
    and overwrites nothing. `tests/test_recovery.py` asserts that the
    displacements are NOT in the record, so if that ever changes the replay can
    be replaced by a read.

65. **2026-09-16, Stage 2c. The evaluation target is fixed, differently on each
    arm, and both are reported beside the old in-sample score.** `[AUTHOR]`
    The old target was the variable-weighted empirical CDF of the same data the
    model was fitted to, which is circular twice over: it is the training data,
    and it is the variable-weighted CDF, which makes "variable weighting improves
    fit" close to true by construction.

    SYNTHETIC: W1 against the KNOWN PARENT. EMPIRICAL: cross-validated W1, ten
    random half-splits in both directions, all six methods sharing each split so
    the comparison is paired. Discrepancy entry 53.

    **TWO PARENT COMPARISONS, AND CONFLATING THEM IS AN ERROR THIS STAGE MADE AND
    CAUGHT.** `w1_parent` scores each method against the parent IT is estimating,
    the sampling mixture for a uniform-weighted method and the market-weighted
    mixture for a variable-weighted one. That is the only fair way to judge an
    ESTIMATION method and it **cannot compare the two weighting schemes**,
    because they are then scored against different truths. `w1_market` scores all
    six against the market-weighted parent, which is the population a pLCA of
    what gets built is a statement about, and is the only target under which the
    weighting question is answerable.

    **A CROSS-VALIDATED SCORE MAY NEVER BE COMPARED ACROSS WEIGHTING SCHEMES.**
    The empirical weights are an exchangeable flat Dirichlet draw, so the
    expected variable-weighted CDF of a random half IS the unweighted one and a
    uniform-weighted fit is the better predictor by construction. That is a
    property of the synthetic weights, not a finding. The weighting claim
    therefore rests on the synthetic arm's market parent and the family claim on
    both arms.

66. **2026-09-16, Stage 2c. The two arms disagree about the family out of sample,
    and the disagreement is explained rather than averaged away.** `[AUTHOR]`
    Both differences survive a paired bootstrap over datasets. Against the parent
    the KDE beats the lognormal by 0.0078 (uniform) and 0.0060 (variable);
    cross-validated on the empirical arm the lognormal beats the KDE by 0.0313
    and 0.0335.

    **They are being read on different criteria and different size mixes.**
    Uniform weighting, removing one at a time: parent, equal allocation +0.0078;
    the same corpus CROSS-VALIDATED instead -0.0034, because a cross-validation
    half measures the KDE at n/2 and its advantage is a large-n advantage;
    reweighted to the empirical size mix -0.0127; the empirical arm itself
    -0.0321. **The criterion and the size mix account for the SIGN.** A factor of
    about two in magnitude does not, and that is a genuine corpus-to-arm
    difference.

    **What every arm and every criterion agrees on is the SHAPE: the KDE loses at
    n = 10-99 and wins at n >= 1000.** The paper states the size dependence and
    does not state a single winner. Entry 54.

67. **2026-09-16, Stage 2c. Every headline aggregate is reported twice, equal
    allocation and reweighted to the empirical size mix.** `[AUTHOR]`
    `coverage.post_stratified` had existed since Stage 2a and no stage had
    applied it to the W1 or rank results. It changes the sign of the corpus's
    family verdict on the MEAN -- `Lognormal, Uniform` 0.1306 against
    `KDE, Uniform` 0.1228 equal allocation, becoming 0.1240 against 0.1268
    reweighted -- while the KDE stays ahead on mean RANK, 2.25 against 2.75.
    Neither column is the true one and both are reported.

    `genconfig.EMPIRICAL_STRATUM_SHARE` was stale: measured on the 149-dataset
    arm, before decision 61. Corrected to 20 / 78 / 38 / 8 over 147, which moves
    `coverage.post_stratified` in notebook 1 by at most 0.27 percent relative.
    `recovery.empirical_size_shares` MEASURES the shares from the arm it is
    given, so the score tables cannot inherit a stale constant again. Three
    empirical datasets exceed the corpus maximum of n = 9,999 -- the three
    largest `ReadyMix` strength classes -- so the reweighting covers 144 of 147.
    Entry 62.

68. **2026-09-16, Stage 2c. The empirical headline is a WIN SHARE, and no
    size-banded claim below about n = 100 stands without the relative-gap view
    beside it.** `[AUTHOR]` Both constraints come from the weight-draw noise
    floor Stage 2b measured: W1 between the same values under two independent
    Dirichlet draws has a median of 0.1344 on the empirical arm against a best
    method's median W1 of 0.0984, so the target's own noise exceeds the best
    score. A mean rank averages a signed distance in rank space and inherits the
    noise of every dataset; a win share is a count, and the noise has to flip a
    dataset's winner to move it. `tests/test_recovery.py` pins the mechanism.

69. **2026-09-16, Stage 2c. W1 stays the criterion. Overlap area is the reported
    robustness check and a tail-sensitive companion is NOT added.** `[AUTHOR]`
    Two measurements settle it.

    **Overlap area agrees.** On the synthetic arm, where a reference density
    exists, it picks the same winner as W1 on 66.6 percent of datasets, correlates
    at Spearman 0.689, and gives the SAME mean-rank ordering of all six methods.
    It is not adopted because it needs a density: the empirical target is a set of
    atoms, and supplying one would mean choosing a bin width or a kernel, and a
    kernel would score the KDE against a KDE. The carried-forward item is closed.
    Entry 58.

    **The tail W1 does not see is now empty, and that is the guard's doing.**
    Integrating each fitted model's survival function beyond the recovery grid
    gives a mean charge of 0.0000 to 0.0001 across 60,000 fits, no fit whose
    unseen tail exceeds its body score, and a rank correlation of 1.0000 between
    the body score and the total. The pathology of decision 51 -- a model with a
    standard deviation of 3,281 on data whose own is 0.6 -- was produced at
    `PROFILE_DELTA_LO_FRAC = 0.01` and does not occur at 0.25. **So W1 alone is
    sufficient AS LONG AS THE GUARD HOLDS, and `model_sd_ratio` stays as the
    sentinel. Stage 2h sweeps that guard and must report `model_sd_ratio` with
    every value it tries.** Stage 2g inherits the same question from the pLCA end.

70. **2026-09-16, Stage 2c. The three-parameter lognormal keeps its place, and
    the paper says plainly that on real data it buys nothing over gamma.**
    `[AUTHOR]` Out of sample on the empirical arm it is indistinguishable from
    gamma, from the two-parameter lognormal and from the Stage 1 offset method:
    every paired bootstrap interval straddles zero. On the synthetic arm against
    the parent it does separate from gamma, +0.0117 uniform and +0.0045 variable,
    winning 77.4 and 67.5 percent of datasets. It is never worse, so it stands.

    **Stage 2b's claim that gamma beats the lognormal ON THE GUARD-BOUND DATASETS
    is withdrawn.** Out of sample, on those 65 empirical datasets, gamma wins 47.7
    percent -- a coin flip -- and on the `interior` datasets it wins 61.4 percent,
    the opposite direction. **No hybrid estimator**, which is what the
    measurements favor least. Entry 59.

71. **2026-09-16, Stage 2c. The bandwidth was re-examined against the parent. It
    confirms Scott is wrong and does NOT confirm the guard, and nothing was
    changed.** `[AUTHOR]` Decision 54 chose the guarded rule on leave-one-out
    likelihood, a DENSITY criterion, while the study scores CDFs. The synthetic
    parent gives the study's own criterion a target that is not the training data.

    **The referee is unbiased**, which had to be established first: only 1.2
    percent of datasets put the optimum at the sweep floor of 0.02 of Scott's,
    against in-sample W1 minimizing there for 95 of 147. Its optimum is at 0.46
    (uniform) and 0.56 (variable) of Scott's.

    **It confirms the direction.** Scott sits 1.386x and 1.349x above the
    parent-optimal bandwidth and the guarded rule beats it on 72.1 and 66.4
    percent of datasets. **It does not confirm the GUARD**: pure Silverman beats
    the guarded rule on 90.1 and 83.2 percent. The guard costs 0.9 and 0.25
    percent of mean W1 against the parent and buys the repaired p05 of held-out
    likelihood it was chosen for. **Said and left alone, per the stage
    instruction; changing the guard is an author decision and Stage 2h owns the
    sweep.**

    The reconciliation belongs in the text: a density criterion and a CDF
    criterion want different bandwidths, because the empirical CDF is already
    root-n consistent so smoothing buys a CDF criterion very little. That is the
    honest explanation, not "W1 is biased". Entry 60.

72. **2026-09-16, Stage 2c. The 1,000-point scoring grid stays as it is.**
    `[RECOMMENDED]` Measured against a 200,001-point lattice it costs a median of
    0.11 to 0.20 percent of a score and picks a different winner on 1.36 percent
    of empirical and 0.50 percent of synthetic datasets. It is biased BY METHOD
    and AGAINST the KDE, by 4.7 percent on `KDE, Variable`'s empirical mean
    against 0.2 percent for the lognormals.

    **It does not reach a conclusion, and that is what decides it.** The
    discretization is common to all six methods on a dataset, so it moves the
    level of every score and not the gap between two of them: the paired
    cross-validated KDE-minus-lognormal difference is -0.0340 at the study's grid
    and route, -0.0341 integrating the same grid as two CDFs, and -0.0339 at
    20,000 points. **One free improvement is NOT taken**: on the same 1,000
    points the CDF route has a p99 relative error of 2.5 percent against the atom
    route's 14.5 percent, at the same cost, but switching moves every reported
    number for no change in any conclusion. Author decision. Entry 61.

73. **2026-09-16, Stage 2c review. The variable-weighting penalty below n = 100
    is the FLAT DIRICHLET STAND-IN, not weighting, and the paper's weighting
    claim changes accordingly.** `[AUTHOR]` Raised by the author against the
    Stage 2c draft: "variable data is parent distribution + noise, so it's not a
    faithful representation of the parent." Correct, and measurable.

    A synthetic dataset's weights are built in two steps. Mode k gets its true
    market share, which is SIGNAL, because the market-weighted parent is a real
    population object at `mode_coupling = 1.0`. That share is then split among
    the points inside mode k by a flat Dirichlet, which is NOISE the real world
    does not have: a market share is a property of a product, not a random draw.

    `audits/weight_noise_vs_signal.py` refits everything under ORACLE weights --
    the same mode-level share, split equally within each mode -- which isolates
    the noise. Paired against the same uniform fit, against the market parent, at
    n = 10-99: KDE **-0.0398 to -0.0056**, lognormal **-0.0335 to -0.0014**,
    normal -0.0235 to -0.0066, and none of the oracle figures is distinguishable
    from zero. At n >= 100 the oracle makes variable weighting BETTER still. At
    n = 3-9 a real penalty survives for the lognormal, -0.0213, because
    estimating a several-mode market mixture from three to nine points does not
    work however clean the weights are.

    **So the claim is NOT "variable weighting hurts below n = 100".** It is that
    variable weighting pays whenever the market shares are actually known, from
    about n = 10 upward, and the penalty this study measures is the price of
    representing UNKNOWN shares with a flat Dirichlet. **This is the strongest
    argument in the project for the real production volumes of Marsh, Hattam and
    Allen (2025).** Entry 64.

    It also needed a code change: the per-point mode label cannot be recovered
    from anything on disk, because `MixtureParent.sample` shuffles the points so
    that mode membership carries no positional information.
    `corpus._replay_one` now returns it.

74. **2026-09-16, Stage 2c review. The KDE's loss at n = 10-99 is real, and the
    three explanations that would have made it an artifact are all excluded.**
    `[AUTHOR]` The author pressed on it for the fourth time in the project, which
    is why it was answered by measurement rather than argued.

    NOT the halving: at fit fractions 0.5, 0.7, 0.8 and 0.9 the empirical deficit
    is -0.0670, -0.0684, -0.0663, -0.0665, so the 50/50 split is the protocol
    most favorable to the KDE of the four. NOT the evaluation protocol at all:
    against the known parent, fitting on every value, it is -0.0172 uniform and
    -0.0228 variable. NOT the over-dispersion: the fitted KDE's spread is 1.19x
    the data's at n = 10-99, but correcting it exactly moves the deficit only
    from -0.0138 to -0.0126 and beats the plain KDE on 47 to 55 percent of
    datasets, a coin flip. The guard costs the KDE about 40 percent of the
    deficit and does not cause it: under pure Silverman the gap is -0.0085
    instead of -0.0138.

    **The mechanism is the ordinary bias-variance tradeoff.** A parametric family
    converges at root-n and a KDE at n^-2/5, so at small n the lognormal's shape
    bias costs less than the KDE's variance, and at large n the bias stops
    shrinking while the variance does not. **The crossover is at n of about 100,
    which is where it is observed.** With 30 points a bumpy nonparametric
    estimate of a smooth truth loses to a smooth three-parameter one however its
    variance is scaled. `audits/cv_fit_fraction.py`,
    `audits/kde_variance_correction.py`. Entries 65 and 66.

75. **2026-09-16, Stage 2c review. `SILVERMAN_MIN_NEFF` stays at 30, and the
    reason is that moving it would be tuning on the reported criterion.**
    `[AUTHOR]` The author asked the right question -- if the guarded rule loses
    to pure Silverman against the parent, adjust the threshold rather than
    accepting it. Swept over 0, 5, 10, 15, 20, 30, 50, 100, 200 and infinity on
    BOTH criteria, `audits/guard_threshold_sweep.py`.

    **The two criteria disagree.** Held-out likelihood peaks at 20 to 30 on both
    arms; W1 against the parent peaks at **5**, where mean W1 is 0.1355 against
    pure Silverman's 0.1367 and the current 30's 0.1400. **So the parent referee
    argues against this THRESHOLD, not against the guard, and Stage 2c section
    4.8 was too broad in saying otherwise.**

    Keeping 30 is the defensible choice: moving the threshold to improve W1 would
    be tuning the setting on the criterion the study reports, which is exactly
    what decision 54 avoided and what makes the bandwidth choice answerable to a
    reviewer. The whole span 20 to 30 differs by about 1 percent of either
    criterion. **The guard itself stays by author decision.** Entry 67.

76. **2026-09-16, Stage 2c review. The size strata stay at 3-9, 10-99, 100-999
    and 1000-9999 with equal allocation.** `[AUTHOR]` The author asked whether
    the buckets should instead reflect the empirical size distribution, and
    worried it would punish the KDE.

    It would, and not by bias. The empirical share above n = 1,000 is 5.6
    percent, so matching it would put about 560 corpus datasets in that band
    instead of 2,500 and widen every interval there by `sqrt(2500/560)` = **2.1**
    -- in the band where the methods differ most and where the KDE's advantage
    lives. Equal allocation plus post-stratification already delivers both
    readings from one corpus, and it is reversible where a different allocation
    would not be, since generation is closed by decisions 47, 48 and 55. **No
    change.** Decision 67 already reports every aggregate both ways.

77. **2026-09-16, Stage 2c review. The bandwidth documentation said "Scott below
    the threshold" and that is not what runs.** `[DELEGATED, chose to fix]`
    `silverman_guarded` is `0.9 * scale * n_eff ** -0.2` THROUGHOUT, with only
    the SCALE guarded: the robust `min(sd, IQR/1.34)` at or above 30 effective
    observations, the plain standard deviation below it. Scott carries 1.06 where
    this carries 0.9, so describing the fallback as Scott overstates the
    small-sample bandwidth by 18 percent. `customstats.weighted_bw`'s main
    docstring had it right; its parameter note and `fitting.BW_METHOD`'s comment
    did not, and a Methods section written from either would have described a
    different method. Both corrected. Entry 68.

78. **2026-09-16, Stage 2c review. The "95 percent of ECC datasets are visibly
    unimodal" figure is a BANDWIDTH, not a property of the data, and the
    generator was tuned against it.** `[AUTHOR]` The author doubted the figure on
    sight -- "I remember seeing a lot of irregularities" -- and was right.

    `modality.n_modes_visible` counts local maxima of `gaussian_kde(x)` at
    scipy's DEFAULT bandwidth, which is Scott's rule. Stage 2c independently
    established that Scott oversmooths this data by about 35 percent. A mode
    counter at an oversmoothing bandwidth undercounts modes. Rescaled: the
    empirical unimodal share is **94.6 percent at scipy's default, 73.1 percent
    at 0.74 of it, and 55.4 percent at 0.6**.

    **The worse half is that the ARM-TO-ARM AGREEMENT is also a property of the
    bandwidth, and decision 38 steered generation by it.** Total variation
    between the arms is 0.0123 at Scott, which is the number the corpus was
    matched on, and 0.0714 at the corrected bandwidth. The gap is datasets with
    three or more visible modes: **8.5 percent of the empirical arm against 1.3
    percent of the corpus.**

    **WHICH WAY IT CUTS: against the KDE, not for it.** Multimodality is the one
    structure a kernel estimate represents and a three-parameter family cannot,
    and the corpus has six times fewer strongly multimodal datasets than the arm.
    The review's worry that the corpus was built to flatter the KDE is the
    opposite of what this shows.

    **Not fixed, and it is an author decision**, because correcting a tuning
    target implies a retune and generation is closed by decisions 47, 48 and 55.
    The recommendation in the handoff is to correct the REPORTED figure, state
    the bandwidth dependence, and NOT regenerate, on the grounds that the gap
    understates the case for the paper's own method. `audits/visible_modes_bandwidth.py`,
    entry 70.

79. **2026-09-16, Stage 2c review. Decision 73 is NARROWED: the oracle-weight
    experiment says less about reality than its first write-up claimed.**
    `[AUTHOR]` "I'm not sure how much I trust weight_noise_vs_signal.py. Still
    feels like noise. I can't think of a way to introduce variable weights to
    equally weighted values sampled from a distribution without it producing
    noise."

    The scepticism is correct. In this generator the within-mode split of a
    mode's market share is uninformative BY CONSTRUCTION -- share attaches at the
    mode level at `mode_coupling = 1.0`, and every point in a mode comes from the
    same component, so dividing the mode's share among its points cannot move the
    target. The oracle removes variance the generator defined to carry no signal,
    so finding that it carries none is not evidence.

    Real market shares differ in BOTH directions: products within a route are not
    identical, so their shares do carry information; and real shares are far more
    concentrated than a flat Dirichlet -- 63.75 percent for Rest-of-World BOF
    steel against an expected 5.2 percent top share at n = 100 -- and
    concentration cuts the effective sample size, which is the mechanism behind
    the penalty.

    **So the claim is not "variable weighting pays whenever shares are known".**
    It is that the penalty is a property of the weight VECTOR rather than of
    weighting, and **this generator cannot say what real shares would do.** That
    is a limitation of the generator and is the more important finding. Stage 2h's
    `mode_share_alpha` sweep can bracket it; only real production volumes would
    settle it.

80. **2026-09-16, Stage 2c review. `SILVERMAN_MIN_NEFF` MOVES FROM 30 TO 20. This
    SUPERSEDES decision 75**, which said it stays at 30. `[AUTHOR]` Stated
    explicitly because CLAUDE.md forbids reversing a decision silently, and
    decision 75 was written hours earlier in the same review.

    **What was wrong with 75's reasoning.** It argued that moving the threshold to
    improve W1 against the parent would be tuning the setting on the criterion the
    study reports. It would not: the reported criterion is IN-SAMPLE W1, and the
    parent score is an independent out-of-sample truth. The author pushed back on
    exactly that point and was right.

    **WHY 20 AND NOT 10, which is the question a reviewer asks.** Step the
    threshold down one value at a time and measure what each step buys in parent
    accuracy per unit of held-out likelihood it costs. Every step from 200 down to
    20 is free or better than free -- 30 to 25 buys 0.65 percent for 0.29, and 25
    to 22 and 22 to 20 cost nothing at all. **The step 20 to 18 is the first that
    costs more than it buys**, at a marginal ratio of 0.34, and every step below
    it is also below 1. The held-out p05, which is the failure the guard exists to
    repair, says the same from the other side: flat at about -1.62 from 200 down
    to 18, then -1.65 at 15, -1.72 at 10, -1.88 at 5.

    So 20 is the smallest threshold reachable by steps that each cost nothing on
    the criterion the guard protects. It is not a round number and it is not the
    W1 optimum, which is 5 and would cost 17 percent of that p05.
    `audits/guard_threshold_sweep.py`, entry 67.

81. **2026-09-16, Stage 2c review. The scoring grid becomes 20,000 points with
    TRAPEZOID quadrature, because the atom route never converges.** `[AUTHOR]`
    "I'm fine with 20,000 atoms - I was worried about computation time but if that
    actually converges and is more accurate, let's do it." It does not converge,
    and that condition is what selected the route.

    W1 is the area between two CDFs. The study evaluated the model's DENSITY at
    1,000 grid points, treated them as weighted atoms, and took the discrete
    Wasserstein distance to the data. The alternative evaluates the model's CDF on
    the same points and integrates the absolute difference.

    **NEITHER ROUTE IS SIMPLY BETTER AT 1,000 POINTS, and an earlier draft of this
    claimed otherwise and was corrected by a test that failed.** Against a
    400,001-point reference, relative error median / p99 / max: atoms
    0.00134 / 0.1359 / 1.385, trapezoid 0.00218 / 0.0228 / 0.200. The atom route
    is better TYPICALLY, because the data's empirical CDF is a step function that
    a discrete-to-discrete distance handles exactly, and far worse in the tail.

    **THE DECIDING FACT: the atom route has a floor it cannot get below.** Adding
    points does not extend the grid, whose top is `max(x) + 10 sd` whatever the
    point count, so a model with mass beyond it keeps losing that mass. Its p99
    relative error sticks at 0.0379 from 20,000 points through 100,000, while the
    trapezoid route goes 0.0039 to 0.0010 to 0.0002. The paired in-sample
    KDE-minus-lognormal difference saturates at -0.0218 for atoms against a true
    -0.0229.

    **THE CHANGE FAVORS THE METHOD THE PAPER IS ABOUT, so it is justified on the
    convergence table and nothing else.** The coarse grid inflated the KDE's score
    by 3 to 5 percent against 0.2 percent for the lognormal, because the KDE's CDF
    has the most structure at grid scale. The in-sample paired difference moves
    from -0.0184 to -0.0229 (uniform) and -0.0216 to -0.0278 (variable). **The
    CROSS-VALIDATED comparison does not move, -0.0340 against -0.0339**, so no
    out-of-sample conclusion of Stage 2c depends on it. `fitting.W1_ROUTE`,
    `SCORE_GRID_POINTS`, entry 61, `audits/scoring_grid_error.py`.

82. **2026-09-16, Stage 2c review. Decision 78 is NARROWED: the reported
    unimodality figure is wrong, and the CORPUS IS FINE.** `[AUTHOR]` Decision 78
    said the corpus under-represents multimodal datasets by a factor of six and is
    therefore biased against the KDE. That was measured at an arbitrary 0.74
    multiple of scipy's default bandwidth and does not survive a better choice.

    **Measured at the bandwidth the study ACTUALLY FITS** -- `silverman_guarded`
    at the threshold of 20 settled by decision 80, which is the density a reader
    is shown and the pLCA samples from, and which needs no invented multiple --
    one visible mode is **68.46 percent empirical against 76.23 synthetic**, two
    modes 26.15 against 20.46, three or more 5.38 against 3.31, total variation
    0.0777. The corpus does under-represent the multi-humped datasets, by a factor
    of 1.6 rather than decision 78's 6.

    **AND IT DOES NOT MATTER: reweighting the corpus to the empirical mode mix
    changes the KDE-minus-lognormal difference by 0.0004 under uniform weighting
    and 0.0002 under variable.** That is the measurement that decides it, not the
    size of the mismatch. `audits/modality_reweighting.py`.

    **So generation is NOT reopened**, decisions 47, 48 and 55 unchanged, and the
    author's offer to rebuild the corpus is declined on the evidence. What stands
    from decision 78 is the part about the REPORTED figure: 95 percent unimodal is
    a property of Scott's bandwidth, it is about 74 percent at the fitted
    bandwidth, and the manuscript must say which bandwidth it is quoting.
    `modality.n_modes_fitted` is the figure to report; `n_modes_visible` stays as
    the generator's tuning target with a docstring saying so. Entry 70.

    Worth keeping from the mode split: **modality helps the KDE without being
    necessary to it.** It is closest on 74 percent of visibly unimodal datasets,
    80 percent of bimodal and 89 percent of those with three or more, so its
    advantage is present in the unimodal majority and merely larger where there
    are several humps.

83. **2026-09-16, Stage 2c review. The comparison is reported BY MATERIAL as well
    as by dataset, and the tiers are fixed on names before any result is seen.**
    `[AUTHOR]` "I think it's worth picking apart which materials are most relevant
    for embodied carbon (structural materials) and which aren't."

    Weighting categories by n was considered and rejected by the author in the
    same message: it would weight by how many EPDs a manufacturer happened to
    publish, which is a property of the market's paperwork rather than of a
    building, and it correlates with the dimension the KDE wins on.
    `src/materialclass.py` splits instead by what a material IS, in the standard
    hot-spot ordering: the structural frame and its binders, the envelope, then
    everything else. **It reads only the category NAME**, and
    `tests/test_materialclass.py` drives the whole assignment with no data in the
    room, which is the same constraint decisions 43, 46 and 60 impose on the
    category rules.

    **The result is the paper's strongest statement.** On the 23 structural
    categories with at least 100 EPDs -- 83 percent of every value in the arm, and
    the materials that dominate embodied carbon -- the KDE is closest on **70
    percent** of datasets under uniform weighting, mean cross-validated W1 0.0659
    against the lognormal's 0.0739. On the envelope and on everything else the
    lognormal wins. **The subgroup is not fished**: the tiers were fixed before any
    result was looked at and n = 100 is the crossover established independently.

    It is a stratification and not an importance weight. The principled versions of
    that are Stage 2i's real-building anchor and the pLCA-against-truth of entry
    69. Entry 71.

84. **2026-09-17. THE MATERIAL TIER ADDS NOTHING BEYOND DATASET SIZE. Decision 83
    is NARROWED: size is the mechanism and the tier is a consequence of it.**
    `[AUTHOR]` The author asked why the comparison was being sliced at "structural
    and n >= 100" and whether the tier was a useful distinction at all. It is
    useful for deciding WHERE the result matters and it is not a second mechanism.

    Regressing the per-dataset log ratio `log(W1_KDE / W1_lognormal)` on `log(n)`
    and then adding the tier as a factor: R2 goes from 0.316 to 0.321 under
    uniform weighting and 0.216 to 0.225 under variable, **F = 0.49 and 0.72,
    p = 0.61 and 0.49. The tier explains nothing the size does not.** Within a
    single size band the tier ordering is not even stable, and the counts per cell
    are 2 to 40.

    **Why the tier looked like a mechanism:** structural categories are the
    well-populated ones. Median n is **140 for structure against 52 for envelope
    and 46 for everything else**, and the largest six are all ReadyMix strength
    classes.

    **What the structural datasets actually are**, and it answers "are the others
    just more lognormal": no, they are BETTER BEHAVED IN EVERY WAY. Median
    coefficient of variation **0.307 against 0.828 and 0.736**, skewness 0.917
    against 1.566 and 1.716, excess kurtosis 1.412 against 3.716 and 5.312, and a
    HIGHER Shapiro statistic against both the normal (0.911 against 0.828, 0.791)
    and the lognormal (0.964 against 0.951, 0.931). Concrete and steel are tight,
    nearly symmetric populations with many EPDs; finishes and furnishings are
    sparse, dispersed and heavy tailed.

    **So the claim the paper makes is ONE mechanism with a threshold**: the KDE
    overtakes the lognormal at about **124 EPDs** under uniform weighting and 204
    under variable, on the empirical arm out of sample, and the materials that
    dominate embodied carbon are the ones that clear it. Reporting a
    "structural and n >= 100" cell as though it were a separate finding
    overstates it, and the figure now plots the log ratio against n colored by
    tier rather than binning by tier. Entry 75.

85. **2026-09-17. The scoring grid gains the model's TAIL BEYOND the grid,
    computed analytically.** `[AUTHOR]` "Do we also need to cover more ground
    along the x-axis? Can't we just extend the bounds? That would capture more
    tail." Yes, and this is the version that costs nothing.

    Above the grid's top every data point is behind us, so the empirical CDF is 1
    and the integrand is the model's survival function; the missing term is its
    mean excess above `hi`, taken on a LOG-SPACED extension to the 1 - 1e-10
    quantile. Extending the LINEAR grid instead would need five times the points
    to hold resolution, and resolution is the thing that mattered in decision 81.

    **IT IS NOT SYMMETRIC ACROSS METHODS, which is the reason it matters.** The
    truncated normal and the KDE put EXACTLY ZERO mass above `max(x) + 10 sd`; the
    three-parameter lognormal puts a mean of 1.8e-4 and up to 4.9e-3 there. So
    omitting it under-charged one family and not the others. Adding it raises the
    lognormal's mean W1 by 0.28 to 0.41 percent, leaves the other four unchanged
    to five decimal places, and takes the worst-case relative error against a
    +400 sd reference from 3.1e-2 to 1.6e-3. Verified against that reference: the
    composite reproduces the paired KDE-minus-lognormal difference to the fifth
    decimal where the body-only grid was 2 percent off.

    `fitting.W1_TAIL_TERM`. This is the third change in a row that moves numbers
    in the KDE's favor, each for an independently correct reason, and the paper
    should present all three as one paragraph about taking the criterion to
    convergence rather than as three separate improvements. Entry 76.

86. **2026-09-17. THE HONEST ANSWER TO "WHICH METHOD SHOULD I USE": the KDE is
    the SAFER default, not the more accurate one, and a size rule beats both.**
    `[AUTHOR]` Written after the author named the risk directly -- "I'm falling
    victim to confirmation bias... I want to make sure we're honest about our
    findings rather than cherry picking results" -- and asked for the assumption
    behind the whole study to be tested rather than confirmed.

    **The assumption under test:** that kernel density estimation is the safe
    default because it is flexible enough that a practitioner never has to decide,
    and can be trusted to do a good job everywhere.

    **It is half right, and right on a different axis than it is usually argued
    on.** Comparing POLICIES against the unreachable per-dataset oracle, uniform
    weighting:

    | policy | empirical CV: mean cost over oracle / worst case | synthetic parent: same |
    |---|---|---|
    | always normal | 71.2 pct / 5.38x | 98.1 pct / 58.1x |
    | always lognormal | **4.9 pct** / 2.75x | 23.0 pct / 24.5x |
    | always KDE | 15.3 pct / **1.61x** | **14.9 pct** / **6.88x** |
    | KDE if n >= 100, else lognormal | **4.5 pct** / **1.57x** | **9.4 pct** / 6.79x |

    **What is true: the KDE never fails badly.** Its worst case is 1.61x on the
    empirical arm against the lognormal's 2.75x, and 6.9x against 24.5x on the
    synthetic arm. A practitioner who cannot inspect every category is protected by
    it in a way the lognormal does not protect them.

    **What is NOT true: that it is the most accurate default.** On the empirical
    arm, out of sample, always-lognormal costs 4.9 percent over the oracle and
    always-KDE costs 15.3. The KDE is within 5 percent of the best method on 47.2
    percent of empirical datasets against the lognormal's 66.9.

    **And the best policy is neither**: use the KDE above about 100 EPDs and a
    lognormal below, which costs 4.5 percent over the oracle on the empirical arm
    and 9.4 on the synthetic, and has the lowest worst case of any policy on both.
    **That is the recommendation the paper should make**, and it is a stronger
    contribution than "use the KDE" because it is a rule a practitioner can apply
    without judgment and it is supported on both arms and on both axes.
    `TABLE_PolicyComparison.csv`. Entry 77.

87. **2026-09-17. The tail integral is bounded at the 1 - 1e-6 quantile, and the
    bound is the study's own Monte Carlo.** `[AUTHOR]` "Let's be careful we're not
    projecting out too much. We still want to keep it within bounds that are
    realistic and matter." The check was warranted and the answer is reassuring.

    At 1 - 1e-10 the integral reached a median of 3.3 times the largest observed
    value, p99 158 times, max 297 -- not absurd, but further than anything that
    matters. At **1 - 1e-6** it reaches a median of **1.9 times the data maximum**
    and a p99 of 31, it captures **99.64 percent** of what the wider bound
    captured, and the difference to the reported score is at most **5e-5 relative
    on any single fit**.

    **Why that quantile and not a round number:** in the 10,000-draw Monte Carlo
    the pLCA runs, the chance of ever sampling beyond it is about 1 percent.
    Beyond it a model's tail cannot affect any result this study reports, so
    charging for it would be charging for something unobservable. The bound is the
    reach of the study's own sampler. `fitting.W1_TAIL_QUANTILE`.

88. **2026-09-17. NOTHING BUT DATASET SIZE BELONGS IN THE RULE, and multimodality
    least of all.** `[AUTHOR]` The author asked whether other statistical
    characteristics are significant enough to enter the "use the KDE above 100
    EPDs" rule, and expected multimodality to be a much bigger factor.

    **Two tests, and they agree.** First, how much each characteristic adds to
    predicting `log(W1_KDE / W1_lognormal)` once `log(n)` is in the model.
    **`n_modes` is LAST of eleven on both weightings**: incremental R2 of 0.00002
    and 0.00104, p = 0.95 and 0.69. On the synthetic arm, where 2,415 datasets
    make almost anything detectable, it reaches 0.004 and 0.009 against `log(n)`'s
    own R2 of 0.616 and 0.395.

    Second, and the test that decides it, whether adding a characteristic to the
    RULE lowers the cost over the oracle. It does not. Uniform weighting,
    empirical cross-validated: `n >= 200` 3.56 percent, `n >= 100` 4.46, always
    lognormal 4.87, **`2+ modes only` 6.17**, `n >= 100 or 2+ modes` 7.06. Using
    modality as the selector is WORSE than not selecting at all, and it triples
    the worst case, 2.755 against 1.575.

    **Why the intuition fails.** The KDE's advantage is general shape matching --
    skewness and tail behavior -- not the ability to resolve separate humps, and
    at the bandwidth the study fits, 68 percent of empirical datasets are visibly
    unimodal anyway. Section 4.10 already showed the advantage is present in the
    unimodal majority; this shows modality adds nothing on top of size even where
    it is present.

    **What does predict the gap, and why it is still not in the rule:**
    `w_v_uw_wasserstein` is the strongest single addition and replicates on both
    weightings, incremental R2 0.046 and 0.055, p = 0.004 and 0.003. It is a
    property of the WEIGHT VECTOR rather than of the data, a practitioner can
    only compute it after choosing weights, and it is Stage 2d's quantity. Noted,
    not used.

    **The rule stays one threshold on one number.** Anywhere between 100 and 200
    works and the choice is not sensitive inside that range. **Stage 2f owns the
    full metric reduction; this is a targeted answer and not that model.**
    `TABLE_RuleCandidates.csv`, `TABLE_RuleSelection.csv`. Entry 79.

89. **2026-09-17. "What is the likelihood variable weights matter for THIS
    dataset" is a good question, it is cheap, and it is STAGE 2d's, not 2c's.**
    `[AUTHOR]` The author's framing: the study already samples weights from a flat
    Dirichlet, so every draw is an allocation it considers possible; what
    proportion of those possible allocations differ enough from uniform to matter?
    "Just so someone can understand how safe their assumption of uniform weighting
    is."

    **Why it is 2d's.** The question needs a threshold for "enough to matter", and
    2d owns exactly that: the named relative measure and the flip probability
    calibrated against it. Building the final version on a placeholder threshold
    is the mistake this project has avoided everywhere else.

    **Why it is worth doing.** `audits/weighting_risk.py` is a feasibility probe:
    147 datasets by 300 draws in **7 seconds**, so the full version is hours, not
    days. Against a placeholder threshold of a tenth of the mean, the median
    probability is 0.847 at n = 3-9, 0.675 at 10-99, 0.218 at 100-999 and
    **0.000 above 1,000**; uniform weighting is safe for 36 of 147 datasets and
    almost never safe for 12.

    **AND IT IS THE ONE PLACE IN THIS STUDY WHERE DISPERSION BEATS SIZE.**
    Spearman with the coefficient of variation **+0.693**, with the interquartile
    range +0.621, with log(n) **-0.569**. Every question about WHICH METHOD FITS
    BEST is driven by n (decisions 84 and 88); whether WEIGHTING MATTERS is driven
    by how spread the values are. **The author's own intuition, that a wider
    interquartile range means a higher chance weights matter, is right and is the
    opposite ordering from the rest of the paper.** That contrast is worth stating
    in the text.

    **What 2d produces from it:** a per-dataset statement a practitioner can act
    on -- "for a category like yours, assuming uniform weights has an X percent
    chance of changing which material ranks first" -- reported against dispersion
    rather than against n. It also supersedes the single-realization
    `w_v_uw_wasserstein` characteristic, which is one draw from this distribution
    and is the open item about a single Dirichlet realization moving per-dataset
    metrics a long way. Entry 80.
