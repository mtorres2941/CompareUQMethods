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

  **THE CITATIONS TO USE, supplied by the author 2026-10-05.** Article: Torres,
  M. I., Lupton, R., Marsh, E., Srubar III, W. V., & Allen, S. (2026). Using
  kernel density estimation and the Dirichlet distribution for uncertainty
  quantification of building material emissions. Resources, Conservation and
  Recycling, 234. https://doi.org/10.1016/j.resconrec.2026.109022 Software:
  **https://doi.org/10.5281/zenodo.19246153**, which is the CONCEPT DOI and is
  what the paper cites, by author decision 2026-10-06: it resolves to whatever
  the latest version is, where the version DOI (...154) pins v1.0.0 and goes
  stale. A note on 2026-10-05 recommended the version DOI and was wrong.

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
  LCA." Six uncertainty characterization scenarios applied to a four-option
  staircase comparison, evaluated with four comparative metrics. Two findings
  matter here: the ranking of top-contributing products within a design changes
  depending on the characterization scenario, and they use dependent sampling
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
- **RULE: A MESSAGE THAT DELIVERS A PROMPT OR NAMES A HANDOFF ARTIFACT ENDS
  THERE. ZERO SENTENCES FOLLOW IT.**

  **When this rule is in force.** Any message containing either (a) text the
  author is meant to copy into another window, or (b) the name of a file written
  for a later session -- a stage report, a handoff, a discrepancy entry.

  **What the rule requires.** The prompt block, or the filename, is the final
  content of the message. Nothing is appended after it: no summary, no caveat,
  no list of things to watch for, no "two places I would push on", no "one thing
  worth noting".

  **Procedure. Run this on the draft before sending.**
  1. Locate the last prompt block or handoff filename in the draft.
  2. Delete every sentence that follows it.
  3. For each deleted sentence ask: does a later session need this? If yes, add
     it to the prompt text or to the file, then verify it is there. If no, it
     was noise and is now correctly gone.
  4. Send the message.

  **Five rationalizations produce this error. Each is wrong, for the reason
  given.**
  - *"The next window should pay attention to X."* X is an instruction. Put it
    in the prompt.
  - *"I should flag where my own work is weakest."* That is a section of the
    report, written for the reader who will attack it.
  - *"This is context, not an instruction."* The author cannot distinguish the
    two from outside, so they must treat everything as an action item.
  - *"It is only one short line."* Length is not the fault. Position is.
  - *"The session went well, so a closing summary is a courtesy."* It is not a
    courtesy. It is work transferred to the author.

  **Why a hard rule rather than a preference.** The author moves the message's
  content into a different window by hand. Anything outside the pasted block is
  either lost or must be merged manually, and in both cases it signals that the
  artifact was incomplete. An artifact that needs a spoken footnote is an
  artifact with a defect; fix the artifact.

  **Provenance.** Stated by the author 2026-09-14 and again 2026-10-01, the
  second time after a review prompt was followed by three things to keep in mind
  for it: "A prompt should be final and should have no additional after
  thoughts." Recorded as one of the most frequent errors made on this project.

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

## Stage report specification

**REPLACES the handoff specification, 2026-09-25.** The separate chat window
that reviewed each stage without repository access is retired, so the rule
"write for a reader who has no checkout" no longer describes anyone. The
replacement reader is different in one way that matters and the format changes
with it.

- **Location and name:** `reports/STAGE_REPORT_<id>.md`, e.g.
  report deleted by decision 251. Existing `HANDOFF_stage-*.md` files keep their
  names; nothing is renamed retroactively.
- **Format:** plain ASCII Markdown, US spelling, same as everything else.

### Who reads it

**The author, and a fresh Claude Code window whose only job is to attack it.**
Both have the repository. **NEITHER HAS DONE THE WORK**, and that is the
property the format protects. A session that has just spent six hours on
something does not interrogate the assumptions it was built on; that is a
property of context, not of capability, and it is why the review step survives
the chat window's retirement.

### What changes now the reader has a checkout

- **Every headline claim carries the command that reproduces it.** This is the
  new requirement and it is the most important one. A claim a reviewer can
  re-run in one line is a claim that cannot quietly go stale. Stage 2h shipped
  six audit results computed on a corpus that had been replaced later in the
  same stage, and a reproduce-command beside each would have exposed every one.
- **A decision number may be a citation, because the reader can open the log.**
  The old rule that a citation must never carry the substance still holds: state
  the claim in full, then cite it. A dead pointer is still a dead pointer.
- **"See this file" is now legitimate** where it used to be forbidden, as long
  as the sentence still means something without opening it.

### What does NOT change, and why

- **Numbers appear as TEXT.** The author reads this without opening tables. If
  a claim needs three numbers, the three numbers are in the sentence.
- **Figures are EMBEDDED**, with captions carrying the numbers, because the
  author reviews the report and the figures together.
- **Every headline claim carries a plain-language "so what"** for someone who
  designs buildings and does not read statistics. A claim that cannot be
  restated in plain words has not been understood well enough to publish.
- **PROVENANCE ON EVERY RESULT: which corpus, and which weight rule.** Added
  2026-09-25 after a stage could not answer that question about its own output
  without reconstructing it from file timestamps.

### The shape

1. **The first page**, and it is capped at one page: what changed, what moved,
   what needs an author decision. This is what the author reads.
2. **The findings**, each with its numbers, its "so what", and its
   reproduce-command.
3. **Numbers that moved**, with before, after and reason. States "none"
   explicitly if nothing moved.
4. **What is still open.** **OPEN ITEMS ONLY, amended 2026-09-27 by the
   author**: "very strange to have things listed as 'closed' under a table
   titled 'what is still open'". An item a previous stage closed belongs in
   the decision log, which is where a reader who wants the history looks; it
   does not get a row here saying it is closed. An item stays only while
   somebody still has to do something about it, and a LIMITATION THE
   MANUSCRIPT STATES is not an open item -- it is a paragraph somebody owes,
   and it goes under what the next stage picks up.
5. **Inputs and outputs**, including which corpus every result ran on.
6. **What the next stage picks up first.**

**AND KEEP IT SHORT, added 2026-09-27 by the author**: "this stage report is
extremely long ... the idea that a single stage report is ~7000 words is wild.
That's about how long the actual manuscript will be." A stage report is a
briefing, not a transcript. Findings, numbers, what needs a decision; the
narrative of how a defect was found belongs in the commit message and the
decision log. Aim well under 2,500 words.

**The test before a stage ends is unchanged in substance:** could the author
read this alone and know what happened, and could a reviewer who has not done
the work find what is wrong with it. If either answer is no, the report is what
needs fixing.

### The review step, which is not optional

At the close of every stage, before the next one starts: open a FRESH window,
give it the stage report and `reports/STAGE_PROMPTS.md` and nothing else at
first, and ask it to find what is wrong, stale, internally contradictory or
asserted without measurement. It may then open the repository to check, but it
forms its questions from the report first. Its standing questions are in
`reports/START_HERE.md`.

### Retention

**Only the CURRENT stage's report is kept.** Earlier ones are deleted when the
project closes: this repository is published alongside the paper and a reader
has no use for the editing process that produced it. What survives is the
decision log (`reports/DECISION_LOG.md`) and `reports/MANUSCRIPT_discrepancies.md`. Git history
retains the deleted files, which is sufficient.

**This does not relax the rule that nothing outstanding may live only in a
conversation.** When a stage closes, its unresolved items move into the decision
log or the discrepancy file, not into a file that is going to be deleted.

---

## How the repository works

Figures are governed by **FIGURE_STYLE.md**, which is binding on every figure
this repository produces: one message per figure with the takeaway in the title,
Tufte's data-ink discipline, direct labeling rather than legends, and a
checklist. `src/figstyle.py` implements what can be implemented. It was written
in Stage 2d at the author's instruction, after a review found the stage's figures
followed no written guide.

Mechanics live in **CONTEXT.md**: package layout, the fitting interface, the
seeding and caching conventions, how to run the pinned environment and the
smoke configuration, the input and output tables, the regression fixture
inventory and what each one pins, and the test suite. Read it before touching
code. It was split out of this file at the end of Stage 1, when this file grew
past a comfortable size.

---

## What this rework retires from the manuscript's Discussion

**A checklist for the manuscript rewrite, assembled in Stage 4 from the decision
log.** Several Discussion paragraphs in the current draft exist to excuse
limitations that no longer hold. Each row names the limitation as drafted, what
replaces it, and the decisions that did it, so the rewrite reads a table rather
than reconstructing this from memory. **Every one of these INVERTS: what was a
caveat is now a measured finding.**

| The draft says | What replaces it | Decisions |
|---|---|---|
| **Only two parametric families were tested** | Six data-driven families were. The three-parameter lognormal is never worse than gamma out of sample and Weibull is the weakest of them; nothing in the conclusions moves when either is added. And the comparison a reader actually cares about is drawn explicitly: against the TWO-parameter lognormal, which is ecoinvent's default and what the pedigree matrix produces, the kernel estimate is 31 to 41 percent closer to the truth above 100 declarations and wins on 71.5 percent of datasets | 70, 151, 167, 189 |
| **Only one KDE bandwidth rule was tried** | Four were, on three criteria. Scott, Silverman, Silverman guarded by a minimum effective sample size, and a cross-validated bandwidth, judged on held-out likelihood, on W1 against the known parent, and through the pLCA to the answer. The shipped rule stands; Scott is worst on the mean of five pLCA outputs under both weightings, and the manuscript's own configuration used Scott, so its numbers UNDERSTATE the kernel estimate -- the conservative direction | 54, 71, 80, 188, 195, 233, 242 |
| **Only one lognormal fitting method was tried** | Five. A three-parameter fit with the threshold chosen by profile likelihood under a calibrated guard, a two-parameter MLE, gamma, Weibull, and the Stage 1 offset method -- plus every parametric family REFITTED by direct W1 minimization, which is the fair-comparison control that answers "the parametric families were judged by a rule they were never fitted under". Against that control the kernel estimate still wins two times in three | 51, 52, 70, 167, 181 |
| **There is no out-of-sample evaluation** | There are three, and the in-sample score is retained only for comparison. The synthetic arm is scored against the KNOWN parent the data were drawn from; the empirical arm is cross-validated over ten random half-splits shared by all six methods; and every pLCA is run a second time against the datasets' TRUE parents, so the error a method causes is measured and not inferred | 65, 66, 109 |
| **The pLCA construction is designed to amplify the effect** | It is, and the amplification is now MEASURED rather than conceded. Group size is swept from 2 to 12 materials and material use intensity over the simplex: a material must lead the next by about 2.1 to 2.2 times before the choice of method cannot change which one leads, and the one real building element available sits at 1.02. Equal intensities are therefore an upper bound on how often a modeling choice changes a real answer, and the bound is quantified | 101, 106, 107, 113 |
| **Goodness-of-fit stands in for what the method does to the answer** | It no longer has to. Every claim a probabilistic LCA makes is scored directly against the truth, and the gap between the two levels is itself a result: a fit threshold of about 81 declarations becomes a claim threshold near 1,000, because a pLCA picks one method for all four of its materials | 143, 163, 166, 174 |

**AND THREE LIMITATIONS THE REWORK ADDS, which the Discussion now owes.**
The corpus spans the modality of real categories and their dispersion and
under-represents their INTERSECTION: multimodal-and-dispersed is 1.3 percent of
the corpus against 5.4 percent of the real arm, so the paper cannot speak to
about one real category in twenty (decision 203). The characteristic on which
the two arms sit furthest apart is now how lognormal they look, not dispersion,
so the limitation paragraph must be rewritten around it (decision 197). And
market shares are simulated rather than observed on both arms, which makes
every weighting number a statement about a share model; on the synthetic arm
the group-level share is the TRUE one, so that arm alone measures what
ignoring a known share costs (decisions 199, 212).

---

## Repository conventions

- Branch per stage: `stage-<id>-<short-name>`, e.g. `stage-0-1-refactor`.
- Commit in logical units so any result change can be bisected.
- **Every session that commits PUSHES its branch before it ends, and confirms
  `git ls-remote origin <branch>` matches `git rev-parse HEAD`.** Added
  2026-10-07, after GitHub turned out to hold nothing past 2026-06-02 (decision
  255). Claude owns this; the author should never have to ask.
- **No tracked file over 50 MB**, and no single push over about 1.5 GB: GitHub
  refuses files over 100 MB and pushes over 2 GB. A large vector figure is
  rasterized by `figstyle.savefig`; a large table goes to Parquet.
- **`main` moves only at a release** -- journal submission or acceptance -- as
  a fast-forward from the working branch, tagged, with a new Zenodo version.
  The working branch is pushed and visible throughout; `main` stays at the
  version the current Zenodo DOI describes.
- `refs/` is gitignored: it holds copyrighted publisher PDFs and large
  third-party datasets, kept locally only.


## Decision log and project history

The numbered decision log (entries 1 onward) and the pipeline roadmap of the
analysis stages, all complete, are in **`reports/DECISION_LOG.md`**. It is
not loaded automatically. Open it to check a specific fact or the reason for
a choice, cite entries as "decision N", and append new decisions there,
newest last. **Never use it as a source for manuscript prose**: the paper
reads as a final report, not a record of corrections.
