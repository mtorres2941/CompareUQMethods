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
decision log in this file and `reports/MANUSCRIPT_discrepancies.md`. Git history
retains the deleted files, which is sufficient.

**This does not relax the rule that nothing outstanding may live only in a
conversation.** When a stage closes, its unresolved items move into the decision
log or the discrepancy file, not into a file that is going to be deleted.

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
| **2c DONE** | The evaluation target: scored the synthetic arm against the known parent (recovered by replaying the generator, decision 64), cross-validated the empirical 147, the fit-versus-definitional decomposition, regret, post-stratification, overlap area, the gamma question, the bandwidth against the parent, and the scoring grid. report deleted by decision 251 | The pLCA construction (2e) and the flip-probability threshold (2d). It did NOT split the uniform-to-variable W1 into location and shape, which is 2d's |
| **2d DONE** | Decomposed the uniform-to-variable W1 into location and shape, named the relative measure and verified it un-normalized, built the per-dataset weighting risk and the size-and-dispersion law behind it, calibrated the flip probability, and measured what switching UQ method does to every pLCA output in real units. report deleted by decision 251 | Building companion decision metrics (2g), and installing common random numbers in the STUDY's pLCA, which stays 2e's |
| **2e DONE** | pLCA construction: common random numbers installed in the study's own pLCA, the crossed sweep over group size and material use intensity, resampled groupings, bootstrap intervals on every headline percentage and NRMSE, the flip thresholds recomputed at every group size, and the pLCA against the TRUE parents. Decisions 105 to 113. report deleted by decision 251 | Changing what the headline metric is (2g). It did NOT redesign the metric set, and it did not touch the fitting, the corpus or the empirical arm |
| **2f DONE** | Shapiro-Wilk versus Shapiro-Francia resolved in favor of Francia under both weightings, `_royston_pvalue` corrected, then the multivariate reduction of the characteristic set against TWO targets -- the fit score and the downstream error -- with the survivors, the size confound, the three modality measures, the definitional tautology, partial dependence, and the curves that replace the rolling averages. **The author's review then moved the whole reduction out of sample and onto the synthetic arm, removed every p-value, and measured the practitioner threshold rather than choosing a tolerance for it.** Decisions 125 to 142. report deleted by decision 251 | Regenerating, redesigning figures (3), and re-running the sweeps of 2h. It did NOT touch the fitting, the scoring criterion, the corpus's values or the empirical extract |
| **2g DONE** | The metric set, judged against the run to the TRUE parents rather than on stability: the rank-1 frequency recovers worst of seven candidates and exceeds its own between-material spread under a normal; three different methods lead across the seven; the normal's 40 percent penalty is 44 percent on attribution and 1 to 4 percent on the tail and information metrics; the `(1-capecc)` divisor settled by dividing by the APPLICABLE iterations and reporting the applicability; two magnitude companions, one of them new; the tail failure mode measured rather than assumed; and the last use of the retired in-sample target removed. Decisions 143 to 150. report deleted by decision 251 | Re-running the sweeps of 2h. It did NOT touch the corpus, the fitting, the scoring criterion or the weight model |
| **2h DONE** | Eighteen sweeps, each closing a "you only tested one variant" objection. **ITS TWO LARGEST RESULTS REVERSE PREMISES THIS TABLE USED TO CARRY.** First, the two arms drew market shares by different rules on the dimension the paper is built on; that is fixed, both arms now use one rule at coherence 0.5, and the tenfold disagreement above 1,000 declarations is a factor of 1.9 (decisions 178, 190). Second, **the dispersion-versus-weighting trade this row called structural was an ARTIFACT of that weighting mismatch and disappears once it is repaired** -- so the corpus was regenerated as `corpus_2026-09-25` with the dispersion distance 0.409 to 0.247 and the weighting distance 0.340 to 0.151, improving together for the first time (decisions 193, 197). Not one recommendation moved: the practitioner threshold is still 81 declarations and every size band has the same winner (decision 198). Also: the scorecard put on one numerator, the both-fits rule for every published crossing, the judgment arm with the pedigree matrix SOURCED and found to be narrower than real data, the certification credit, Weibull, the bandwidth through the pLCA, the parent-level gate and the end-to-end smoke test. Decisions 174 to 199. report deleted by decision 251 | Anything not framed as a sweep with a tabulated result. It did NOT adopt the upper truncation (decision 199) and did NOT measure the mixed-method policy, which is 2j |
| **2i** (optional) | Real-building anchor, only if we decide after 2g that citing Marsh et al. (in press) is not enough | Becoming a case study |
| **2j DONE** | **THE MIXED-METHOD POLICY, AND THE HONEST VERSION IS MODEST.** The rule a practitioner can follow -- uniform weights throughout, kernel estimate above the cutoff and three-parameter lognormal below -- beats the best uniform-weighted method on 9 of 16 claims, median 0.9 percent, pooled 2.9 percent (decision 217). **The 16-of-16 at 11.4 percent belongs to the same switch PLUS known market shares, which nobody has** (decisions 204, 216), so the gap between the two is the VALUE OF MARKET-SHARE DATA at 12.8 percent -- four times what the rule itself is worth, and the paper's strongest practical statement. **The cutoff is published as 40 to 170** (decision 224), the feasible rule's own indistinguishable band under a paired difference test against a FIXED reference -- the earlier band rule was a contest among grid points and moved when the grid was filled in. The claim-level constant is 80, not 81. A SECOND and much tighter band answers a different question: knowing MARKET SHARE is significantly harmful below 80 declarations and significantly helpful above 100, statistically zero in between, on all 10,000 datasets (decision 222). The sweep's degenerate ends reproduce their fixed methods to 0.00e+00 (decisions 213, 218). **And a correction the paper must carry**: the synthetic arm's market weights are the TRUE group-level shares to 1.1e-16, not a flat-Dirichlet guess, so using a KNOWN share is what hurts below about 81 declarations -- because importance weights re-aim a fixed sample rather than adding to it, leaving a median Kish effective sample of 2.8 at 3 to 9 declarations, and the bandwidth's effective sample size is NOT the cause (decisions 212, 215). Nothing already on disk moved (decision 206); the argmax qualification is dropped (decision 210); Stage 3 adds the FEASIBLE rule as a seventh scorecard column (decisions 211, 214). report deleted by decision 251 | Inventing a second selector; every candidate is still one cutoff on one number. It did NOT reopen `mode_share_alpha`, settled by decision 203 |
| **3 DONE** | Figures. Notebooks 1 and 2 got an `OUT` cell and the compute/plot split that makes their figure cells redrawable, and **the split moved nothing: 147 x 24 empirical values bit identical, four of five PNGs byte identical**. Figures 2 and 3 merged; Figure 4 rebuilt on the 2f survivors with its three-curve alternative beside it; the feasible rule added as a seventh scorecard column; one `savefig` helper writing a PNG and a vector sibling and refusing a duplicate name; the figure manifest and four guard tests; the six stale audits re-run, every ordering holding and decisions 182, 184, 187 and 195 needing updated numbers; the hump-spacing levers measured alone and still declined; the two 96 MB tables moved to Parquet; the ASCII, spelling and vocabulary sweeps. report deleted by decision 251 | Changing any number. **The figure NUMBERING is DEFERRED to the manuscript by decision 235 and is not Stage 4's either**; full FIGURE_STYLE.md compliance for the 29 figure cells that never call the style module is DEFERRED with it, by the same decision, and was never Stage 4's -- an earlier version of this row said it was |
| **4 DONE** | **The four code changes first, then ONE run, then the controls (decision 239), and all three held.** The seventh scorecard column rebuilt on the SAME truth pass as the other six, whose control reads 0.000e+00 on the numerator across all 90 cells -- and the published count is **10 of 15**, not decision 237's forecast of 11, because that forecast was itself a one-pass number from the OTHER pass and the deciding claim is a tie at 13.32 against 13.21 (decision 246). The duplicate claim dropped, denominator 15 (247). The feasible rule's group-composition column, with all four degenerate-end controls at 0.00e+00. All 37 of 37 figure cells renderable, which turned up **a figure cell consuming the notebook's random stream** -- 720,000 draws -- now moved to a compute cell and the figure pixel identical (248). Decision 241 withdrawn: the panel labels predate Stage 3 by five months and the real defect was a label collision (249). Then the deposit: which corpus the paper describes, the README's account of the weights which was wrong in two ways, the figure manifest, the spelling and vocabulary sweeps, and the superseded reports deleted (250, 251). `reports/STAGE_REPORT_4.md` | Anything analytical. **NOT the `.git` history rewrite: declined by the author, decision 28.** **NOT the figure NUMBERING, NOT full `FIGURE_STYLE.md` compliance for the 29 cells that never call `figstyle.apply()`, and NOT confidence intervals on figure aggregates**: all three are per-figure work that waits on the manuscript's figure selection, decision 235 |

Items already known to be open and owned by a named stage, so that none of them
reads as a fresh discovery: the bandwidth rule and its KL1/KL2 inconsistency
(2h); `logfit_offset` (RESOLVED in 2b, retired; the profile-likelihood guard `PROFILE_DELTA_LO_FRAC` is what 2h sweeps in its place); the "Mode Count" label naming a
continuous modality index (2a); the 27.5 percent filter and its n cap at 749
(2a); the variance-inflation exponent and the reflection step (2a);
Shapiro-Wilk versus Shapiro-Francia and `_royston_pvalue` (RESOLVED in
2f, decisions 125 and 126); dependent
sampling (2e); overlap area alongside W1 (2c); the scoring grid's zero
(RESOLVED in 2b); the `(1-capecc)` divisor (RESOLVED in 2g, decision 147: the denominator is the iterations in which the strategy APPLIES, and the applicability is reported rather than assumed constant); overlap area (RESOLVED in 2c, decision 69); W1's lack of a complexity penalty (RESOLVED in 2c, decisions 65 and 70); the linear scoring grid (RESOLVED in 2c, decision 72);
`weighted_quantile` order dependence (fixed in Stage 1 Phase 3, and it must
stay fixed before any switch to Silverman in 2h); the uniform-to-variable
location/shape split and the flip-probability threshold (RESOLVED in 2d,
decisions 92 and 95); `w_v_uw_wasserstein` as a single Dirichlet realization
(RESOLVED in 2d, superseded by the per-dataset risk of decision 94);
common random numbers in the STUDY's pLCA (RESOLVED in 2e, decision 105:
installed, and measured to be worth 4 to 15 percent of the model difference on
every continuous output and a 3.67 percent floor on the argmax); dependent
sampling (RESOLVED in 2e by the same change, which is what dependent sampling
across compared options means); the materials-per-pLCA sweep, resampled
groupings and bootstrap intervals (RESOLVED in 2e, decisions 106, 107 and 110);
the pLCA against the true parents (RESOLVED in 2e, decision 109); the
smoke-run guard (RESOLVED in 2e, decision 111, which was Stage 3's item).

Mark each stage done as it completes. If a stage hands an item to a different
stage than this table says, update the table rather than leaving the two out of
step.

**Outside the stage structure, and belonging to the manuscript session, not to
any analysis stage:** revising the manuscript, and
`reports/MANUSCRIPT_discrepancies.md`, which every stage appends to and which
is worked from later. Keep appending. Do not start editing the paper.

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
   `.docx` carries unresolved comments from a named third party, and git
   history would retain them even after deletion. **Measured 2026-10-05 on
   `CompareUQMethods_BE1_Manuscript_v1_wvs.docx`: 97 comments, all from one
   author, dated 2026-08-21 to 2026-08-31.** The earlier "98 from named third
   parties" overstated both the count and the number of commenters; the rule is
   unaffected and so is the reason for it.
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
    rather than 25 so it cannot be read as a tuned threshold.

    **AMENDED 2026-09-17, Stage 2d: THE ICE FIGURE IS UNSOURCED AND IS WITHDRAWN
    FROM THE PAPER.** This entry asked a later stage to verify it before it was
    printed. Stage 2d checked: `refs/` holds no copy of ICE v3.0 and no file in
    this repository contains the number, so neither the figure, the edition nor
    the page can be confirmed from anything here. **Do not cite ICE.** The
    ceiling stands on the arithmetic alone, which needs no database and which a
    reviewer can check in one line: combusting pure carbon yields
    44.009 / 12.011 = 3.664 kg CO2 per kg of carbon, so 100 kgCO2e per kg of
    delivered product requires burning 27.3 kg of pure carbon for every kilogram
    shipped, and even 25 kgCO2e/kg requires 6.8 kg. **This is about what the
    paper cites, not about whether the ceiling is right**; the ceiling is
    unchanged and no number moves. Discrepancy entry 81.

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
    **NARROWED 2026-09-22 for one tool; see the end of this entry.**
    `[AUTHOR]` "Everything should be traceable back to the notebooks. The
    notebooks should reproduce the entirety of this analysis." Two figures dated
    2026-03 had no producer anywhere in the repository and were deleted; one
    figure had been written by a scratch script and its code is now a notebook
    cell. **Audit scripts may write only under `outputs/tables/audits/`, never
    to `outputs/figures/` or the top level of `outputs/tables/`.**

    **THE NARROWING, 2026-09-22.** `audits/render_figures.py` may write to
    `outputs/figures/`. The author allowed it after a twenty-minute notebook run
    was spent moving a label: "I want all figures to be reproducible in the
    notebooks, but if you have a faster way to reproduce the figure so we can
    iterate, I'm open to it. As long as the notebook reflects those changes."

    **It is a different EXECUTOR for the same bytes, not a second author of
    figures, and that is enforced rather than promised.** The script contains no
    figure code and no analysis code at all: it reads the notebook, executes the
    notebook's own setup cell to get the imports and the output root, then
    executes the notebook's own figure cells verbatim.
    `tests/test_render_figures.py` asserts that the source executed is
    byte-identical to the notebook's and that the module holds no plotting call.

    **It REFUSES rather than skips**, and that mattered immediately: a cell that
    saves a figure without starting with `# FIGURE` or `# SUPPLEMENT` makes the
    tool exit naming that cell, because redrawing some figures and reporting
    success would leave stale figures committed beside fresh ones with nothing
    to say so. An earlier version of the marker matched two of notebook 4's
    three figure cells and would have done exactly that. Notebook 4 is fully
    marked; notebooks 1 and 2 define no `OUT` and notebooks 1 to 3 have unmarked
    figure cells, so the tool declines them until Stage 3's figure work.

    Seven seconds against about twenty minutes, which is the difference between
    iterating on a figure and not.
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

90. **2026-09-17. SUPERSEDED by the author, after Stage 2d measured it: A_IQR is
    DROPPED from this paper.** "That was a hypothesis I had that didn't pan out
    ... let's stop mentioning it and don't bring it up in the manuscript. It
    served a different purpose for a different study." The measure stays in
    `src/weighting.py` and its per-dataset values stay in
    `TABLE_WeightingRisk.csv`; the figure that showed it is deleted and nothing
    in the manuscript cites it. See decision 94 for what was learned before it
    was dropped, and decision 96 for the size-and-dispersion law that replaced
    it. The original entry follows, unchanged, because a later stage needs to
    know what was asked for and why.

    **A_IQR is the measure Stage 2d should use for "how safe is
    uniform weighting", and it comes from the author's own KL2 paper, which makes
    it a CONSISTENCY CONSTRAINT rather than one option among several.**
    `[AUTHOR]` Recorded because it was clarified in conversation and would
    otherwise be lost; decision 89 specified the question and this names the
    instrument.

    **The definition, as the author gave it:** sample many weight vectors from the
    Dirichlet, fit a PDF under each, and take **the area of the interquartile
    range of that ensemble of PDFs**. A higher A_IQR means a wider error band
    around the PDF you would have got from uniform weights. One number per
    dataset, directly answering how much the density wobbles when the market
    shares are unknown.

    **TWO CONSTRUCTION DETAILS ARE SETTLED BY THE AUTHOR, 2026-09-17.** The
    quartiles are taken **POINTWISE IN x** across the ensemble, and the component
    PDFs use the **GUARDED SILVERMAN BANDWIDTH**, `fitting.BW_METHOD`, to align
    with the rest of this study. Note that the second is a deliberate divergence
    from KL2 if KL2 used a different rule: this paper's bandwidth is settled by
    decisions 54 and 80, and a weighting-risk measure computed at some other
    bandwidth would not describe the densities this paper actually fits.

    **TWO DETAILS STILL HAVE TO BE READ OFF THE PAPER.** Torres, Lupton, Marsh,
    Srubar and Allen (2026), RC&R 234, 109022, is in `refs/` as
    `1-s2.0-S0921344926002466-main.pdf`. What remains: **how the area is
    normalized, if at all**, and **how many Dirichlet draws** the ensemble uses.
    Getting either wrong would produce a number that shares a name with KL2's and
    not a definition, which is worse than inventing a new one. **And when the
    paper reports A_IQR it must state the bandwidth**, because this study's is now
    fixed by decision 80 and may not be KL2's.

    **Why it is a constraint and not a choice.** CLAUDE.md's standing rule treats
    the author's two published papers as consistency constraints. A_IQR is KL2's
    answer to the same question this paper asks, so using it keeps the two
    consistent and lets this paper cite rather than re-derive. **Where this paper
    differs from KL2 on it, say so**, as the bandwidth inconsistency of decision 9
    had to be said.

    **How it relates to what Stage 2c measured.** `audits/weighting_risk.py` uses a
    different and more ad-hoc quantity: W1 between the uniform-weighted and the
    variable-weighted empirical CDF, per draw, thresholded. That was a feasibility
    probe, not a proposal. **A_IQR is the better instrument** -- it is a single
    number, it is in density space where a practitioner reads an error band, and it
    is already published. The probe's finding survives the change of instrument and
    is the reason to expect A_IQR to work here: dispersion, not size, is what drives
    whether weighting matters (Spearman +0.693 with the coefficient of variation
    against -0.569 with log n), and A_IQR is a dispersion-of-the-density measure.

91. **2026-09-17, Stage 2d. WITHDRAWN IN FULL. The claim that the study's pLCA
    "cannot answer did the answer change" was wrong, and the 5.33 percent figure
    behind it is not to be quoted.** `[AUTHOR]` This entry originally reported
    that comparing two UQ methods under independent random streams changes the
    answer 5.33 percent of the time, and treated that as a defect in the study's
    pLCA. The author rejected it five times before it was demonstrated rather than
    asserted, and the demonstration shows the objection was correct.

    **What that number actually counts.** Not a change in any reported quantity.
    It is the rate at which the LABEL "which material has the highest rank-1
    frequency" lands elsewhere when two materials are statistically tied. Worked
    case: the four frequencies were 0.2335, 0.2581, 0.2597, 0.2487 in one run and
    0.2325, 0.2583, 0.2543, 0.2549 in another. The largest change in any number is
    0.0062; the top two differed by 0.0016, inside their own 0.0043 standard
    error. **10,000 draws is ample and the numbers agree; an arbitrary tie-break
    went the other way.**

    **Why quoting it was actively misleading.** "5.33 percent of comparisons
    change" reads as though a contribution moved from 25 percent to 20 percent.
    Nothing of the sort occurs. Decision 103 measures what does change, in units a
    reader can act on.

    **The one thing that survives.** The flip-probability calibration of decision
    95 has an argmax as its outcome, so it would inherit that tie-break noise;
    giving both methods the same uniform draws removes it exactly, which is why
    the calibration is run that way. `families.rvs_from_uniform`. Installing the
    same thing in the study's own pLCA is Stage 2e's and is a refinement, not a
    repair.

92. **2026-09-17, Stage 2d. The uniform-to-variable W1 is MOSTLY A SHIFT OF THE
    MEAN, and that simplifies the practitioner rule.** `[AUTHOR]` The author
    asked for the decomposition and named the good outcome in advance: if it is
    mostly location, the rule collapses to something computable in a spreadsheet.
    It is.

    W1 is bounded below by the absolute difference in means, and here the two
    distributions are the same values under two weightings, so the bound is
    `abs(weighted mean - unweighted mean)`. Calling that the LOCATION component
    and the remainder SHAPE:

        arm         median share   mean share   pooled   above half
        empirical      0.7252        0.6726     0.7281     68.7 pct
        synthetic      0.8043        0.6945     0.7992     72.4 pct

    **The share is highest where the datasets are smallest**: on the empirical arm
    it is 0.96 at n = 3-9, 0.65 at 10-99, 0.74 at 100-999 and 0.54 at 1,000 and
    above. With three to nine values there is barely any shape for reweighting to
    change.

    So the guidance a practitioner needs is "compute a weighted mean and see how
    far it moves", not a distributional calculation. The residual is real but
    secondary, and it grows with dataset size, which is the opposite of where the
    weighting question is most urgent. Entry 83.

93. **2026-09-17, Stage 2d. The relative measure is W1 DIVIDED BY THE DATASET'S
    UNWEIGHTED MEAN, it was already there unnamed, and naming it moves no
    number.** `[DELEGATED, 2d chose]` Every dataset in this study is divided by
    its own unweighted mean before anything else happens (decision 6), so every
    W1 this study has ever reported is already a W1 divided by a mean. The
    manuscript nowhere says so.

    **Verified rather than asserted.** The same quantity was recomputed on the
    RAW empirical values, in their own kgCO2e per declared unit, with dataset
    means spanning **0.0685 to 910.7**, a factor of 13,000. The relative measure
    agrees with the normalized run to **6e-15**; the absolute W1 scales by exactly
    the dataset's own mean, which is the control that the check is checking
    something. `tests/test_weighting.py` pins both.

    **The two robust alternatives were computed and are not adopted.** Dividing by
    the interquartile range or the standard deviation instead separates the flips
    essentially as well: AUC on the calibration set is 0.8018 for the mean, 0.8004
    for the interquartile range and 0.8056 for the standard deviation on the
    top-contributor outcome. A difference of half a percent is not a reason to
    change the denominator the study already uses, and the mean is the only one of
    the three a practitioner computes without ambiguity. **The standard deviation
    is better on the FULL ORDERING outcome**, 0.794 against 0.761, which is noted
    and not acted on because the full ordering is not a criterion this study
    reports.

    **All three are computed with UNIFORM weights, and that is the decision that
    matters here** rather than which of the three is used. A denominator taken
    under the variable weights would move when the weights move, which is the
    quantity being measured, and a practitioner holding a set of EPDs cannot
    compute a market-weighted mean without already knowing the market shares.
    Entry 86.

94. **2026-09-17, Stage 2d. A_IQR IS NEARLY A FUNCTION OF DATASET SIZE AND IS
    ALMOST BLIND TO DISPERSION. This NARROWS decision 90**, which adopted it
    expecting the opposite. `[DELEGATED, 2d measured]` Stated explicitly because
    CLAUDE.md forbids reversing a decision silently, and decision 90 was written
    hours earlier.

    **What decision 90 expected.** The Stage 2c probe found that whether weighting
    matters is driven by dispersion rather than size, Spearman +0.693 with the
    coefficient of variation against -0.569 with log n, and decision 90 adopted
    A_IQR on the reasoning that it is a dispersion-of-the-density measure and
    would inherit that. It does not.

    **THE MECHANISM, AND THE FIRST VERSION OF THIS ENTRY GOT IT WRONG.** It said
    A_IQR cannot see dispersion because it is exactly invariant under rescaling
    the data. It IS invariant -- verified to ten decimal places over seven orders
    of magnitude in `tests/test_weighting.py` -- but **so is the mean-relative
    separation this stage uses instead**, so invariance cannot be what
    distinguishes them. That explanation is withdrawn.

    **What separates them is what each divides by.** A_IQR measures the
    uncertainty of the density curve against that curve's own height and width, so
    the data's spread cancels from both factors and what survives is the sampling
    noise in the weights, which is a question of how many points there are. The
    separation is a distance along x divided by the mean alone, so the ratio of
    spread to mean survives, and that ratio IS the coefficient of variation.

    Both halves check out. Holding a lognormal at n = 60 and raising the
    coefficient of variation from 0.22 to 5.83, a factor of 27: A_IQR goes 0.296,
    0.285, 0.299, 0.307 while `A_IQR * sqrt(n)` stays between 2.21 and 2.38; the
    separation goes 0.030, 0.099, 0.316, 0.664, and **divided by the coefficient
    of variation it is nearly constant at 0.138, 0.119, 0.115, 0.114**.

    **A_IQR is DOMINATED by size, not blind to spread**, and the within-band
    numbers say where the difference lies. Its rank correlation with the
    coefficient of variation inside each size band of the real arm is -0.008 at
    n = 3-9, +0.128 at 10-99, +0.644 at 100-999 and +0.405 above 1,000: monotone
    but small. A five- to tenfold change in the coefficient of variation within a
    band moves A_IQR by a factor of 1.07 to 1.87, against a factor of 12 across
    the size range.

    **THE DISPERSION RESULT SURVIVES ON THE OTHER MEASURE, AND IT IS A
    BOTH-MATTER RESULT RATHER THAN A REVERSAL.** An earlier draft of this entry
    claimed the probe's ordering was reproduced and strengthened; that was read
    off a SATURATED variable and is withdrawn. The thresholded probability sits
    at exactly 1.000 for 46 percent of real categories and above 0.99 for 74
    percent, because the calibrated threshold is far smaller than a typical
    reweighting, so its marginal correlation is carried by the untied minority.

    The honest quantity is the MEDIAN SEPARATION over draws, which has no
    ceiling: **+0.731 with the coefficient of variation and -0.545 with log n**,
    reproducing the probe's +0.693 and -0.569 almost exactly. Both drive it.
    Dispersion is marginally the stronger and that is remarkable in a study where
    dispersion predicts nothing else, but it does not displace size.

    **Where dispersion dominates is WITHIN a size band**, and there it is nearly
    deterministic: Spearman +0.940 at n = 3-9 over 20 datasets, +0.888 at 10-99
    over 78, +0.955 at 100-999 over 38, and +0.833 above 1,000 over 8. The mirror
    holds among the 38 unsaturated categories, where size explains almost
    everything (-0.726) and dispersion almost nothing (+0.040). **So the claim
    the paper makes is that which METHOD fits best is driven by size alone, while
    whether WEIGHTING matters is driven by size across categories and by
    dispersion within a size band.**

    **SUPERSEDED 2026-09-17 BY THE AUTHOR: A_IQR IS DROPPED FROM THE PAPER
    ENTIRELY.** "That was a hypothesis I had that didn't pan out ... let's stop
    mentioning it and don't bring it up in the manuscript. It served a different
    purpose for a different study." The code stays in `src/weighting.py` and the
    per-dataset values stay in `TABLE_WeightingRisk.csv`; the figure that showed
    it has been deleted. Nothing in the manuscript cites it, and decision 90's
    adoption of it is superseded. The practitioner statement is made on the
    distance between the uniform-weighted fit and the fit under a drawn market
    share, in units of the dataset's own mean.

    **Two details were read off KL2 as decision 90 required**, from
    `refs/1-s2.0-S0921344926002466-main.pdf`, and both are now settled. **The area
    is NOT normalized**: the paper reports bare areas of 0.40, 0.22 and 0.12 for
    its three scenarios and applies no divisor, and none is needed because the
    integral is already dimensionless. **The ensemble is 1,000 draws**, stated
    twice -- "Fig. 3a shows 1000 iterations illustrating the range of viable
    solutions" and, for the 131-EPD steel proof of concept, "these constraints are
    incorporated to generate 1000 viable PDFs". This study uses 1,000 for
    consistency. Entry 84.

95. **2026-09-17, Stage 2d. THE CALIBRATION CURVE: what a given W1 costs, and the
    relative W1 at which the top contributor flips 1, 5 and 10 percent of the
    time.** `[AUTHOR]` This is the stage's deliverable.

    Measured on 2,500 pLCA groups by nine tempering levels, under common random
    numbers, with a logistic fit on log distance and a percentile interval from a
    bootstrap that resamples pLCA GROUPS rather than rows:

        level   crossing   95 pct interval      isotonic
        1 pct     0.0018   0.0013 to 0.0023       0.0026
        5 pct     0.011    0.0091 to 0.0126       0.0129
        10 pct    0.025    0.0217 to 0.0277       0.0271

    The isotonic fit assumes only that the flip probability does not FALL as the
    models separate, so its agreement says the crossings are a property of the
    data and not of the link function. `flip.FLIP_THRESHOLDS` carries them.

    **TWO SIGNIFICANT FIGURES, AND NO MORE.** The interval is about 30 percent of
    the estimate wide, and an independent run of the same calculation on a
    different random stream gave 0.00149, 0.00991 and 0.02334 -- every one inside
    the intervals above, and every one differing in the third figure. Quoting
    five would be false precision and would make the constant drift on every
    rerun, which matters because notebook 1 reads it to turn a per-dataset
    weighting risk into a probability. Notebook 3 prints the recomputed crossing
    beside the stored constant on every run so that drift stays visible.

    **THE SIX UQ METHODS COULD NOT SUPPLY THIS CURVE AND THAT IS ITSELF A
    RESULT.** Over 37,500 comparisons the smallest relative W1 between any two of
    the six is 0.00022, and the flip rate in the lowest 2 percent of separations
    is already 14.1 percent. Every
    level being asked about lies below the observed data. So the calibration set
    adds pairs at controlled separations running to zero -- the same kernel
    estimate under uniform weights and under weights moved a fraction `t` of the
    way toward a Dirichlet draw -- with `t = 0` as the control and `t = 1` as the
    study's own variable weighting.

    **The device is checked, not assumed.** If the curve describes the DISTANCE
    rather than its provenance, the six real method pairs, which are different
    FAMILIES, fall on the curve fitted from weighting pairs. They do over most of
    the range: 0.109 against 0.129, 0.180 against 0.166, 0.280 against 0.287 in
    three successive separation bands. **They diverge above a separation of
    0.122**, 0.429 against 0.604, so a cross-family difference of a given size is
    more consequential than a reweighting difference of the same size and the
    curve UNDERSTATES the flip probability there. Said rather than smoothed.

    **THE CAVEAT TRAVELS WITH EVERY ONE OF THESE NUMBERS.** Every material here is
    normalized to a mean of 1.0 with a material use intensity of 1.0, so the four
    contributions are nearly exchangeable and the ranking is as fragile as it can
    be made. A real building, where materials differ by orders of magnitude, is
    much harder to flip. These are an **upper bound** on how often a modeling
    choice changes an answer, which is the conservative direction for a
    practitioner rule and must not be quoted as a statement about buildings.

    **The full rank ordering is NOT a usable criterion in this construction** and
    is reported only to say so: its crossings are at 0.00002, 0.00023 and 0.00065.
    Ranking four near-identical materials from first to last is not a decision
    anyone makes. Entry 85.

96. **2026-09-17, Stage 2d. WHETHER WEIGHTING MATTERS HAS A CLOSED FORM IN TWO
    NUMBERS A PRACTITIONER ALREADY HAS, and it is the strongest
    practitioner-facing result in the study.** `[AUTHOR]` Asked for directly:
    "Is there a way we can visualize the contribution of size and dispersion?"
    The answer turned out to be better than a figure.

    Regressing the log of the separation between the uniform-weighted fit and a
    Dirichlet-weighted one on log size and log dispersion: **size alone explains
    48.0 percent of the variance, dispersion alone 49.5 percent, and both
    together 99.1 percent.** Each adds about half on top of the other, so they
    are nearly orthogonal, which is exactly why neither looked like the answer on
    its own.

        empirical   log(sep) = -0.318 - 0.434 log(n) + 1.036 log(CV)   R2 0.991
        synthetic   log(sep) = -0.405 - 0.427 log(n) + 0.977 log(CV)   R2 0.996

    **The exponents agree across the two arms**, which is what makes this a law
    rather than a fit: **separation is about 0.73 * CV * n^-0.43**.

    **The rule needs no distributional machinery at all.** Against the calibrated
    5 percent flip threshold of 0.011, uniform weighting is safe only when the
    coefficient of variation is below about **0.015 * n^0.43**: 0.046 at 10 EPDs,
    0.120 at 100, 0.315 at 1,000, 0.826 at 10,000. The median real category is CV
    0.63 at 47 EPDs and does not clear it. A reader counts their EPDs and takes a
    coefficient of variation in a spreadsheet; every other rule this project has
    produced requires the analysis to have been run first.

    **The contrast the paper must draw.** Which METHOD fits best is driven by
    size and nothing else, and modality was tested and rejected for that role
    (decision 88). Whether WEIGHTING matters is driven by size AND dispersion. A
    reader who has absorbed the first will carry it into the second and be half
    wrong. Discrepancy entry 88.

97. **2026-09-17, Stage 2d. THE FLAT DIRICHLET UNDERSTATES THE WEIGHTING RISK,
    so every weighting number in this paper is a LOWER BOUND.** `[AUTHOR]` The
    author's question: a flat Dirichlet explores the simplex uniformly, but real
    market share probably arrives in clusters; and if it does, is a cluster not
    almost a dataset with fewer points, so that the effective sample size already
    captures it?

    **Half of that is right and the other half is the important half.** Three
    weight schemes compared at MATCHED Kish effective sample size, 97 categories
    by 30 draws: share concentrated on RANDOMLY chosen products gives 0.84 to
    1.32 times the flat separation, which is no difference at all -- **so the
    effective sample size does capture concentration, exactly as the author
    reasoned.** Share concentrated on products with ADJACENT coefficients, which
    is what clustering means in practice, gives **1.5 to 3.1 times** the
    separation at the same effective sample size.

    **Concentration and coherence are different things and only the first is a
    sample-size effect.** A contiguous block shifts the whole distribution one
    way, and that lands in the LOCATION term which decision 92 shows already
    carries a median of 72.5 percent of the uniform-to-variable distance; random
    concentration moves mass in directions that partly cancel.

    **The direction is conservative, which is what makes it a limitation rather
    than a hole.** If real shares cluster by product similarity -- and the 63.75
    percent share Marsh, Hattam and Allen (2025) report for Rest-of-World BOF
    steel says they do -- the true separations are larger and uniform weighting
    is even less safe than this study reports. Only real production volumes would
    settle it. **Stage 2h's concentration sweep should vary the BLOCK STRUCTURE
    and not only the Dirichlet concentration parameter**, because the two are not
    the same knob. `weighting.block_weights`, discrepancy entry 89.

98. **2026-09-17, Stage 2d. Silverman's divisor stays at 1.34, and the paper
    states in one sentence that KL2's 1.35 is the same rule.** `[AUTHOR]` The
    author asked which is correct. **Neither is wrong and 1.35 is the closer
    rounding.** The interquartile range of a standard normal is
    `2 * 0.674490 = 1.348980`, so the unbiased divisor is 1.3490: dividing by
    1.35 biases the scale estimate by -0.08 percent and by 1.34 by +0.67 percent.
    1.34 is what Silverman's 1986 book prints and what most software carries.

    The two differ by 0.75 percent in the resulting bandwidth, against the 18
    percent that separates Silverman's coefficient from Scott's, so nothing in
    this study turns on it. **The code does not change**, because 1.34 is the
    published convention and moving it would shift every KDE number for a
    correction nobody asked for. Discrepancy entry 90.

99. **2026-09-17, Stage 2d. THE FLIP THRESHOLD IS CONDITIONAL ON FOUR MATERIALS
    and Stage 2e will move it.** `[AUTHOR]` The author asked why the calibration
    still uses groups of four when the plan was to stop doing that, and whether
    the sweep should have come first. The concern is legitimate and the answer is
    that the machinery does not depend on the group size while the numbers do.

    **The crossings of 0.0018, 0.011 and 0.025 hold for four materials of equal
    material use intensity.** With more materials competing there are more
    chances for a near-tie, so the flip probability at a given model distance
    should RISE and the thresholds should FALL. Stage 2e owns the sweep over 2 to
    12 materials and must re-run `flip.weighting_calibration` at each size,
    reporting the crossings as a function of it; if they move materially, every
    weighting-risk probability in notebook 1 is recomputed. Until then the paper
    states the conditionality. Discrepancy entry 91.

100. **2026-09-17, Stage 2d. An INDUSTRY-AVERAGE EPD may be the missing weighted
     mean, and it is KL2's `EVtarget` under another name.** `[AUTHOR]` The
     author's idea, and it is a good one: since the uniform-to-variable distance
     has a median location share of 0.725, what is missing is a market-weighted
     mean, and an industry-average declaration is in principle exactly that -- a production-weighted average over a population they cannot see.

     **It is already in the author's own published work.** KL2 uses
     industry-average ECCs as its target expected value and places a phantom
     kernel so the model reproduces it, so this paper would be reaching for an
     instrument its companion paper defines, which is the consistency this
     project requires rather than a new invention.

     **Three things must be checked before it is used**: whether the industry
     average is production-weighted at all rather than a simple mean over
     participating manufacturers, which is a different object; what population it
     covers, since a regional average cannot stand in for a global one; and
     whether its scope, boundary and reference year match the declarations it
     would be compared against.

     **The use is as a CHECK, not a replacement.** The distance between the
     unweighted mean of the EPDs a practitioner holds and the industry-average
     value directly estimates the location term, which is most of the effect,
     and needs no Dirichlet sampling at all. **Not currently owned by any stage.**
     Discrepancy entry 92.

101. **2026-09-17, Stage 2d. Every material carrying a material use intensity of
     1.0 makes the ranking maximally fragile, and that biases every result in the
     conservative direction.** `[AUTHOR]` The author asked whether the assumption
     should be challenged. It should, it is scheduled, and in the meantime it has
     to be stated rather than assumed away.

     All four materials are normalized to a mean of 1.0 and weighted equally, so
     they contribute the same expected amount and the ranking is decided entirely
     by the tails. That is why the flip probabilities are high and the
     calibrated thresholds are small. **A real building has materials differing by orders of magnitude, so
     every flip probability this study reports is an UPPER BOUND** on how often a
     modeling choice changes a real answer.

     Stage 2e owns the dominant-intensity variant and Stage 2i the optional
     real-building anchor. **The paper must not quote a flip probability as
     though it described a building**, and the conditionality belongs in the same
     paragraph as the number, not in a footnote. Discrepancy entry 93.

102. **2026-09-17, Stage 2d review. THE HEADLINE SHOULD BE A CONTINUOUS OUTPUT,
     NOT AN ARGMAX. "56 percent of pLCAs flip" is demoted.** `[AUTHOR]` The
     author declined to over-index on the flip rate, on two grounds: the pLCA
     process is still changing in later stages, and the identity of the largest
     contributor is not the only thing a probabilistic LCA is for. Both are
     right, and the measurement supports a better headline.

     **The author's framing of what a pLCA is for:** magnitude and variance --
     which material contributes most, and how likely that is -- because those are
     what drive design decisions and data-collection priorities.

     **The study already computes the right quantities and the error was
     collapsing them to an argmax.** `eci_rank_1` per material IS magnitude and
     likelihood together: "a 30 percent chance this material is the largest
     contributor." What is fragile is not that number but the question "which
     material has the highest one", because with four exchangeable materials the
     top two are often tied. Measured over 300 groups with the SAME models and
     two streams, the argmax flips 12.3 percent of the time for contribution
     share, 5.3 percent for rank-1 frequency and 3.0 percent for variance
     importance, while the underlying continuous values are stable to about 7
     percent of their spread across materials.

     **So the headline becomes a SHIFT rather than a FLIP**, and it needs no
     common random numbers and carries no noise floor:

         choosing between the CLOSEST pair of the six methods moves a
         material's rank-1 frequency by a median of 0.048
         the FURTHEST pair moves it by 0.188

     Read as: which UQ method you pick changes your stated probability that a
     given material is the largest contributor by **5 to 19 percentage points**.
     That is a sentence a practitioner can act on, it is stable, and it does not
     depend on the four-material construction the way a flip rate does.

     **The flip rate stays as a secondary, clearly conditional statistic.**
     **Stage 2g owns the metric set** and should carry this through: report the
     continuous shift as primary, add variance importance as the
     data-collection-facing measure, and state any argmax result with its noise
     floor beside it. Discrepancy entry 94.

103. **2026-09-17, Stage 2d review. WHAT SWITCHING UQ METHOD ACTUALLY DOES,
     in units a reader can act on. Decisions 91 and 102's framing is WITHDRAWN
     along with the statistic behind it.** `[AUTHOR]` The author asked for this
     test -- compare each method pair's model distance against the difference it
     makes in every output -- and separately rejected, five times, a statistic
     this stage kept quoting. Both are settled here and the author was right on
     both.

     **What was withdrawn, and why it was wrong.** Stage 2d reported that running
     one method twice on different random draws changed "the answer" 5.33 percent
     of the time, and built a case on it. That number is the rate at which the
     LABEL "which material has the highest rank-1 frequency" lands on a different
     material. It is not a change in any reported quantity. A worked case: the
     four rank-1 frequencies were 0.2335, 0.2581, 0.2597, 0.2487 in one run and
     0.2325, 0.2583, 0.2543, 0.2549 in the other -- the largest change in any
     number is 0.0062, and the top two differed by 0.0016, inside their own
     0.0043 standard error. **Nothing meaningful moved; a coin-toss tie-break
     landed the other way.** Quoting it as a percentage invited the reading that a
     contribution had shifted from 25 percent to 20 percent, which is not remotely
     what happens. **It is not to be quoted again**, and any ratio expressed as a
     multiple of it is equally meaningless to a reader.

     **What replaces it.** `audits/output_metric_sensitivity.py`, over 250 pLCA
     groups and all fifteen method pairs, both methods given the same uniform
     draws. Each figure is the change for the most-affected of the four
     materials, which is the one a practitioner is deciding about. Every material
     contributes a mean of 1.00, so these are readable directly:

         output                              median   90th pct
         estimated contribution               0.181      0.519
         95th percentile of contribution      0.413      1.290
         standard deviation of contribution   0.173      0.467
         coefficient of variation             0.175      0.389
         chance of being largest contributor  0.126      0.278
         contribution to total variance       0.089      0.276
         share of the building total          0.030      0.085

     **Switching UQ method changes a material's estimated contribution by about
     18 percent.** That is the sentence the paper can use.

     **Two findings inside it.** The SPREAD outputs move most -- the 95th
     percentile by 0.41 against the mean's 0.18 -- which is the right way round,
     because representing spread is what a UQ method is for. And the CONTRIBUTION
     TO TOTAL VARIANCE moves least, so **"where should I collect better data" is
     the most robust output a probabilistic LCA produces**, more robust than any
     magnitude it reports. The study computes it as `ui` and reports it nowhere.
     **CORRECTED 2026-09-22, Stage 2g: "reports it nowhere" IS FALSE.** The
     manuscript reports the uncertainty index in Figure 5b and 5d, defines it in
     Supplement 3(c) and draws a conclusion from it in the results and in the
     conclusions. What is true is that it is reported as a secondary observation
     about NRMSE rather than as one of the questions a probabilistic LCA
     answers, so the instruction is to PROMOTE it, not to introduce it. Decision
     146, discrepancy entry 145.
     Discrepancy entry 95.

104. **2026-09-17, Stage 2d review. The location share is NOT universal, and
     decision 92's "about three quarters" was a median quoted as though it were a
     constant.** `[AUTHOR]` The author asked where a universal 0.725 came from,
     and whether there are categories where reweighting changes the shape rather
     than the mean. There are, and the spread is wide.

     Across the 147 real categories the location share has a median of 0.725 but
     an interquartile range of **0.457 to 0.933** and a full range of 0.024 to
     1.000. **28 of 147 categories are SHAPE-DOMINATED**, with a location share
     below 0.4: `CementGrout` at 0.024, `PowerCabling` at 0.077,
     `CMU [>=6000 psi]` at 0.079, `RebarSteel` at 0.082, `AluminiumExtrusions` at
     0.118. For those, reweighting changes the shape of the distribution and
     barely moves its mean.

     **And it is not predictable from anything.** Spearman of the location share
     with log dataset size is **-0.113** and with the coefficient of variation
     **-0.100**. So unlike the separation itself, which decision 96 shows is
     almost fully determined by those two numbers, HOW the weighting acts is a
     per-category property that neither predicts.

     **The practical framing in decision 92 was also wrong** and the author named
     it: "nobody has the information to compute a weighted mean. Weights for EPDs
     aren't published." Correct, and it is the premise of the companion paper. The
     finding is not a recipe; it is a statement about what KIND of uncertainty
     unknown weights introduce -- mostly uncertainty about the mean, for most
     categories -- which is why an industry-average EPD, the one published
     quantity that IS production-weighted, would resolve most of it (decision
     100). **The practitioner-facing rule is decision 96's size-and-dispersion
     law, not this.**

105. **2026-09-18, Stage 2e. THE pLCA NOW USES COMMON RANDOM NUMBERS, and the
     measurement says it was a refinement rather than a repair, exactly as the
     author argued.** `[AUTHOR]` One uniform variate per material per Monte
     Carlo iteration, drawn once for the group and pushed through every
     method's inverse CDF. Independent ACROSS MATERIALS within an iteration,
     because materials in a building are not rank-correlated; identical ACROSS
     METHODS, so two identical models produce identical results. The practice
     Henriksson et al. (2015) and Heijungs (2021) recommend for comparative
     probabilistic LCA and that Marsh et al. (in press) use. The
     capped-reduction strategy is paired too, through a per-group cache of
     redraw variates.

     **WHAT IT WAS WORTH, over 300 pLCAs and all fifteen method pairs.** Each
     figure is the change for the most-affected of the four materials, and
     every material contributes a mean of 1.00:

         output                    paired   unpaired   noise   noise/paired
         estimated contribution    0.1819    0.1828   0.0095      0.053
         95th pct of contribution  0.4152    0.4153   0.0239      0.058
         chance of being largest   0.1275    0.1274   0.0073      0.057
         contribution to variance  0.0898    0.0914   0.0132      0.147
         TOP CONTRIBUTOR CHANGES   0.548     0.559    0.0367      0.067

     **The unpaired column is the study's old measurement and it was not
     inflated**: every ratio of unpaired to paired is between 0.99 and 1.02. The
     sampling noise is 4 to 15 percent of the model difference and adds nearly
     orthogonally to it, so it never moved a median. **Decision 103's
     withdrawal of the 5.33 percent statistic is confirmed from the other
     side**: the thing common random numbers fix is an ARGMAX, where 3.67
     percent of comparisons name a different top contributor with NO model
     difference at all, and that floor is now exactly zero by construction.

     **NUMBERS THAT MOVED.** Every row of `TABLE_PLCAResults.csv` moves by Monte
     Carlo noise, and no aggregate moves. Per row, `eci_rank_1` changes by a
     mean of 0.0047 and at most 0.0288, which is 4.4 percent of that column's
     standard deviation; the largest relative move in ANY column's mean across
     the whole 60,000-row table is 0.34 percent, on `capecc_rank_4`, whose mean
     is 0.0039. `eci_mean` moves from 1.042702 to 1.042689. The integer rank
     columns move most, by a mean of 0.25 of a rank, which is the argmax
     fragility above and not a change in any quantity. Seven figures drawn from
     that table are redrawn. **The Stage 2d flip calibration is BYTE IDENTICAL**,
     because its streams are spawned from the seed sequence rather than taken
     from the consumed stream, which is the check that nothing else moved.

106. **2026-09-18, Stage 2e. MORE MATERIALS DOES NOT DILUTE THE EFFECT OF
     CHOOSING A UQ METHOD. Only the quantities that are shares of the whole
     dilute, and the probability that the ranking changes RISES.** `[AUTHOR]`
     The author's expectation, stated in the stage prompt, was that the effect
     would dilute as the group grows because each material's share of the total
     shrinks. Half of that is right and the important half is not.

     Equal intensities, 400 resampled groupings per size, 15 method pairs each:

         materials per pLCA          2      3      4      6      8     12
         P(top contributor changes)  .448   .544   .546   .624   .634   .656
         change in a material's
           estimated contribution    .069   .076   .081   .089   .089   .095
         change in its share
           of the building total     .0199  .0204  .0179  .0137  .0108  .0077
         change in its chance of
           being largest             .0684  .0790  .0727  .0585  .0476  .0362

     **The share of the building total does dilute**, by a factor of 2.6 from
     two materials to twelve, and so does a rank-1 frequency, which is bounded
     by 1/k on average and must. **A material's own estimated contribution does
     not dilute at all**; it grows slightly, because a bigger group has more
     chances to contain a dataset the methods disagree about. And the
     probability that the two methods name a different largest contributor
     RISES from 45 to 66 percent, because more materials means more chances of a
     near-tie at the top.

     So the paper's statement has to name the output: **switching UQ method
     changes a material's estimated contribution by about the same amount
     however many materials the building has, and changes its share of the
     total less as the building grows.** All intervals are cluster bootstraps
     over pLCA groups and are about 0.02 wide on the flip rates.

107. **2026-09-18, Stage 2e. A MATERIAL MUST LEAD THE NEXT BY ABOUT A FACTOR OF
     2 BEFORE THE CHOICE OF UQ METHOD CANNOT CHANGE WHICH ONE LEADS, and the one
     real building element available sits at 1.02.** `[AUTHOR]` This is the
     intensity sweep's deliverable and the answer to the question Stage 2d's
     thresholds left conditional.

     **Sampled on the simplex, not in absolute units**, because a building total
     is arbitrary and material use intensity matters only through each
     material's share of the total mean contribution. A symmetric Dirichlet over
     the group's materials, concentration from 200 down to 0.15, plus
     deterministic checkpoints at 1:1, 2:1, 10:1 and 100:1. An infinite
     concentration reproduces the equal case EXACTLY, which is the sweep's
     anchor to the study's own construction and is asserted by a test.

     **Reported against the ratio of the largest mean contribution to the second
     largest**, which a practitioner computes from a quantity take-off in one
     line, and never against the Dirichlet concentration, which means nothing to
     a reader and whose implied dominance changes with the number of materials.

         P(the UQ method changes the leader)   crosses at a ratio of
         1 percent                             2.13   [2.09, 2.17]
         5 percent                             1.64   [1.62, 1.66]
         10 percent                            1.46   [1.45, 1.47]

     An isotonic fit, which assumes only that the probability does not rise as
     the leader pulls away, gives 2.22, 1.61 and 1.35.

     **THE THREE CROSSINGS PRINTED ABOVE ARE SUPERSEDED AND MUST NOT BE
     PUBLISHED. They were measured on `corpus_2026-09-21` and the pre-port
     empirical weight rule, both replaced by decisions 190 and 197.** On the
     shipped corpus the 1 percent crossing at FOUR materials is **2.28 against
     2.33** (parametric against monotone), 5 percent 1.7 against 1.6, 10
     percent 1.53 against 1.51; pooled over group sizes the 1 percent crossing
     is 2.5 against 2.3. The SHAPE of this entry -- that the crossing moves
     little with group size, and that dominance protects the ranking without
     moving the magnitude -- is unchanged. See decision 252 and take the
     numbers from `TABLE_PLCARatioCrossings.csv`, which carries both fits and
     the interval. **The crossing moves
     little with the group size**: the 1 percent crossing is 1.90 at two
     materials and 2.34 at twelve.

     **AND A DOMINANT MATERIAL DOES NOTHING FOR THE NUMBERS.** At the study's
     own four materials the median change in a material's estimated contribution
     is 0.188 at equal intensities, 0.186 at 2:1, 0.175 at 10:1 and 0.166 at
     100:1 -- an 11 percent decline while the flip probability goes from 0.546
     to zero.

     **THE CLAIM IS ABOUT A FIXED GROUP SIZE AND MUST BE STATED THAT WAY.** The
     change in the MOST-AFFECTED material is a maximum over the group, so it
     grows with the group whatever the intensities: 0.109 at two materials and
     0.346 at twelve, at equal intensities. Within a group size dominance moves
     it very little and not always downward -- at twelve materials it runs
     0.346, 0.338, 0.381, 0.433 across the same four cases. What falls cleanly
     everywhere is the change in the AVERAGE material, from 0.081 to 0.045 at
     four materials and 0.095 to 0.046 at twelve, and as a share of the whole
     building the most-affected material's change is flat at 0.03 to 0.05
     throughout. **A figure that pools the group sizes shows a rise, because a
     Dirichlet draw over twelve materials reaches a large top-two ratio far more
     often than one over two, so the right of the axis fills with large groups.
     That is a group-size effect wearing a dominance label**, and the figure is
     held at four materials for it. **That contrast is the figure
     and it is the finding**: concentration protects the RANKING and leaves the
     RESULT where it was.

     **The comparison is in absolute units and that is legitimate here**, because
     the intensity vector is normalized to a mean of 1.0 in every cell, so the
     building's total mean contribution is the number of materials whatever the
     concentration. Decision 113 says what changes if the denominator is the
     leading material's own contribution instead.

     **THE ANCHOR, and it is the reason this matters.** Marsh, Lewis, Hattam and
     Allen (in press) report for their Concrete-Precast staircase under the ICE
     recommended factors that the top two products are steel bar at 42 percent
     and precast concrete at 41 percent, **a top-two ratio of 1.02**. A real
     building element can sit within a percentage point of a tie, which is
     exactly where the choice of UQ method decides the ranking. Their
     per-product quantities are in supplementary material this repository does
     not hold, so no further ratio is computed and no bill of quantities is
     reconstructed; for the other three factor sources the paper states only
     that precast concrete leads at 65 percent, which bounds the ratio without
     identifying it.

     **THE SCOPE LIMIT, which belongs in the text.** Material use intensity is
     deterministic within a run here. Real quantity take-offs carry their own
     uncertainty, which in practice can exceed the coefficient uncertainty this
     paper is about. That is a limitation to state, not a sweep to add.

108. **2026-09-18, Stage 2e. THE FLIP THRESHOLDS RISE WITH GROUP SIZE UNDER THE
     MAXIMUM AND ARE FLAT UNDER THE MEAN, so the stored constants stand and
     notebook 1 is not recomputed.** `[AUTHOR]` Decision 99 required this
     measurement and predicted the thresholds would FALL, on the reasoning that
     more materials give more chances of a near-tie. The flip RATE does rise --
     0.106 at two materials to 0.188 at twelve -- but the thresholds do not
     fall, and the reason is an aggregation artifact worth stating.

     The model distance is a property of a MATERIAL and the decision is a
     property of the GROUP, so the per-material distances have to be summarized.
     The stored constants use the MAXIMUM, and a maximum over twelve materials
     is drawn from more chances than a maximum over two, so it drifts upward on
     its own. Both aggregations, 300 resampled groupings per size:

         materials      2      3      4      6      8     12
         max,  1 pct   .0015  .0020  .0012  .0021  .0022  .0032
         max,  5 pct   .0098  .0108  .0087  .0126  .0124  .0157
         mean, 1 pct   .0011  .0012  .0006  .0010  .0010  .0011
         mean, 5 pct   .0065  .0060  .0041  .0052  .0047  .0049

     **Under the mean the thresholds are constant in the group size.** Under the
     maximum they roughly double from two materials to twelve, and that is the
     summary drifting rather than the pLCA changing.

     At four materials the recomputed crossings are 0.0012, 0.0087 and 0.0216
     against the stored 0.0018, 0.011 and 0.025, which is the same two **[SUPERSEDED CONSTANTS -- the published values are 0.0029, 0.015 and 0.032; see decision 223.]**
    
     significant figures given that these use 300 resampled groupings against
     the stored values' 2,500 and that decision 95 already records a 30 percent
     interval width. **`flip.FLIP_THRESHOLDS` is unchanged and notebook 1's
     weighting-risk probabilities are not recomputed.** The paper states the
     conditionality with the measured dependence beside it.

     **THE CONSTANTS IN THIS ENTRY ARE SUPERSEDED. `flip.FLIP_THRESHOLDS` carries 0.0029, 0.015 and 0.032 from the close of Stage 2h; the values printed here are the pre-recalibration ones. See decision 223 -- do not publish the numbers in this entry.**

109. **2026-09-18, Stage 2e. THE pLCA AGAINST THE TRUTH: the KDE and the
     lognormal are INDISTINGUISHABLE at the decision level, the normal is 40
     percent worse, and no method recovers the answer.** `[AUTHOR]` Stage 2c
     called this the most valuable single experiment left in the project and
     nobody had run it. Every pLCA group is run twice on the same uniform
     variates, once with the fitted models and once with the datasets' TRUE
     parents, so the difference is the error the fitted model causes with no
     Monte Carlo noise in it at all.

     The truth is the MARKET-weighted parent, because a probabilistic LCA of
     what gets built is a statement about the population weighted by production
     and it is the one population all six methods can be scored against on equal
     terms. 2,500 groups, 10,000 draws, cluster-bootstrap intervals:

         method                error in a material's   error in its estimated
                               rank-1 frequency        contribution
         Lognormal, Uniform    0.0799 [.0781, .0817]   0.1265
         KDE, Uniform          0.0815 [.0797, .0834]   0.1313
         KDE, Variable         0.0825 [.0799, .0849]   0.1262
         Lognormal, Variable   0.0853 [.0827, .0877]   0.1201
         Normal, Uniform       0.1152 [.1132, .1172]   0.1639
         Normal, Variable      0.1193 [.1170, .1215]   0.1683

     **THE FOUR NON-NORMAL METHODS SPAN 0.0799 TO 0.0853, a spread of 7 percent,
     and the normal is 40 percent worse than any of them.** So at the decision
     level the choice between a kernel estimate and a three-parameter lognormal
     does not matter, and the choice to use a normal does. **That is a cleaner
     recommendation than a ranking**, and it is the null the stage was told to
     report if it found one.

     **NOBODY RECOVERS THE ANSWER.** The best method names the material the
     truth says is the largest contributor **52.5 percent** of the time
     (`KDE, Variable`), against 25 percent for a coin toss among four, and the
     normal manages 23.9 to 37.4. On a win share over 10,000 materials the best
     is `KDE, Variable` at 0.221 [0.212, 0.231] against a one-in-six chance of
     0.167. Reweighting to the empirical size mix moves every figure by less
     than 0.003.

     **The uniform-weighted methods look much better against the SAMPLING parent
     -- `KDE, Uniform` 0.0609 against 0.0815 -- and the variable-weighted ones
     worse.** That gap is definitional, not an error of estimation: it is the
     difference between the population a method estimates and the population a
     building is about, and decision 65 is why both are reported.

     **The caveat travels with it.** Every material here carries an intensity of
     1.0, which makes the ranking as fragile as it can be made, so the
     rank-based figures are an upper bound on how often a method gets the
     ranking wrong. The contribution error does not have that dependence.

110. **2026-09-18, Stage 2e. EVERY HEADLINE PERCENTAGE AND EVERY NRMSE NOW
     CARRIES A BOOTSTRAP INTERVAL, and the resampling unit is the pLCA group.**
     `[AUTHOR]` The study reported an NRMSE for every pLCA output and attached
     uncertainty to none of them, which is awkward in a paper about uncertainty.

     `plca.nrmse_ci` pivots the results table once and resamples groups of rows,
     which is what makes an interval on forty outputs affordable. The unit is
     the GROUP because the four materials of a pLCA share its total and its
     variates; a row bootstrap comes back more than twice too narrow and a test
     pins that.

     **The headline: `eci_rank_1` has an NRMSE of 1.042 [1.033, 1.051].** A
     value above 1 means the root mean squared difference between two UQ methods
     exceeds the standard deviation of that output across every material and
     method -- the choice of method moves the answer by more than the spread it
     is trying to describe. The lowest of the main outputs is the uncertainty
     index at 0.503 [0.491, 0.515], which is decision 103's finding from the
     other direction: "where should I collect better data" is the steadiest
     thing a probabilistic LCA says.

111. **2026-09-18, Stage 2e. A SMOKE RUN CAN NO LONGER REACH `outputs/`.**
     `[DELEGATED, 2e chose]` This closes discrepancy entry 87, which Stage 2d
     opened after a smoke run reached a commit and replaced the 60,000-row pLCA
     table with a 960-row one while redrawing seven figures from 40 groups.

     Every path notebook 3 writes goes through `OUT`, which smoke mode points at
     a fresh temporary directory, so the repository is untouched. Verified: an
     8-group run wrote all nine tables and eight figures into a temporary
     directory and left `outputs/` clean. Two tests hold it: no cell of notebook
     3 may write a literal `outputs/` path and `OUT` must be defined before the
     first write; and the committed pLCA table must be a full run, not flagged
     as smoke in its metadata, at least 2,000 groups, with the row count that
     metadata implies.

     The roadmap gave this to Stage 3. It was done here because Stage 2e reruns
     the most expensive artifact in the project and the rule that protects it
     was a sentence in a document.

112. **2026-09-18, Stage 2e. THREE CELLS OF NOTEBOOK 3 HAD NEVER RUN, and one of
     them draws a figure in the paper.** `[DELEGATED, 2e chose]` Found by
     running the notebook end to end under the smoke configuration, which had
     never been done for the cells Stage 2d added.

     The flip-calibration figure read `dct_empirical`, which notebook 3 never
     defines, and called `fitting.fit_kde` when only the names imported FROM
     `fitting` were in scope. Both work in a kernel that has run notebook 2
     first, which is how they were written, and both fail in a headless run.
     Separately, `plca` was a loop index in notebook 3 long before
     `src/plca.py` existed, so importing the module left every later call
     reading an integer.

     **What this cost and why it matters beyond the fix:** decision 56 says
     everything must be traceable back to the notebooks, and
     `CompareUQMethods_FIG_FlipCalibration.png` could not be regenerated from
     them as committed. `tests/test_notebooks.py` now refuses a variable that
     shadows an imported module, and the notebook loads the one empirical
     dataset that figure illustrates, spawning its stream after the calibration's
     so that no Stage 2d number moves.

113. **2026-09-18, Stage 2e. THE TRUTH RUN AND THE DOMINANCE SWEEP AGREE, BY TWO
     INDEPENDENT ROUTES: concentration fixes the RANKING and leaves the ERROR IN
     THE NUMBERS exactly where it was.** `[DELEGATED, 2e measured]` Decision 107
     found this by comparing methods with each other. This finds it by comparing
     each method with the right answer, which is a different measurement and
     could have disagreed.

     The same pLCA run against the true parents at three intensity settings, 600
     groups each, with the leading material at 1, 2 and 10 times every other:

         intensity                       1:1          2:1          10:1
         names the TRUE leader      0.23 to 0.51  0.95 to 0.98   1.00 (all six)
         error in rank-1 frequency  0.078-0.119   0.056-0.076    0.0075-0.0091
         error in contribution      0.115-0.161   0.116-0.163    0.119-0.173

     **At 10:1 every one of the six methods names the true largest contributor
     in every group**, and the error in a material's rank-1 frequency falls by a
     factor of ten. **The error in its estimated contribution does not move at
     all.**

     **WHY THE ABSOLUTE COMPARISON IS THE RIGHT ONE HERE, and it is a property
     of the construction rather than an assumption.** The intensity vector is
     normalized to a mean of 1.0 in every cell of the sweep, so the building's
     total mean contribution is the same number -- the group size -- whatever
     the concentration. An absolute error of 0.12 is therefore the same share of
     the building at 1:1 as at 100:1, and the flat row above is a statement
     about the building and not an artifact of rescaling one material. Read as a
     fraction of the LEADING material's own contribution the same error does
     fall, because that material is larger; the paper should say which
     denominator it is using.

     **What the pair of results licenses the paper to say.** A practitioner
     whose design has one dominant material can trust the ranking under any of
     these methods and still cannot trust the magnitude, and the magnitude is
     what a carbon budget is written in.

114. **2026-09-18, Stage 2e review. THE PAPER IS RE-CENTERED ON THE FIVE
     STATEMENTS A PROBABILISTIC LCA MAKES, and the ranking metrics are demoted
     to one of them.** `[AUTHOR]` The author's objection to the first draft of
     this stage: it led with the error in a material's chance of being the
     largest contributor, and "if something predicts a different material as
     being first, but first and second are extremely close, the fact that one is
     over the other doesn't seem very important". Correct, and the same
     objection retired the flip rate in decision 102.

     Three sections, five statements, nothing dropped:

     **What the method does to the NUMBERS.** The building total as a
     distribution rather than a mean and a standard deviation, including the
     chance of meeting a budget; and each material's contribution and share.

     **What it does to WHERE THE UNCERTAINTY SITS.** The uncertainty index,
     which is the most stable output measured and is reported nowhere.
     **THAT LAST CLAUSE IS FALSE and is corrected by decision 146**: the
     manuscript reports it in two panels of Figure 5, defines it in the
     supplement, and concludes from it that the methods give similar uncertainty
     indices. It is under-reported, not unreported.

     **What it does to THE DECISION.** The two interventions with their
     confidence, and the design swap, which is the comparison this study had
     never made.

     **The fourth reduction strategy needs no code and that is the point.**
     Collapsing a material's uncertainty by obtaining a supplier-specific
     declaration is exactly what the uncertainty index measures. Three
     strategies reduce the expected impact -- use less, specify better,
     substitute -- and the fourth reduces the VARIANCE of the answer. Naming it
     that way is the contribution; the arithmetic already existed.

115. **2026-09-18, Stage 2e review. THE ECC CAP WAS TAKEN FROM EACH METHOD'S OWN
     DRAWS, WHICH FORCED THE STRATEGY'S SIGNAL TO ZERO. It is now one absolute
     value per material.** `[AUTHOR]` The author's question: "If I'm comparing
     reduction strategies on an LCA, each one would have the same absolute cap,
     right?" Yes, and the consequence is worse than a comparability problem.

     The cap was `np.quantile(col, 0.75)` on each method's own 10,000 draws, so
     the six methods were asked about six different interventions -- and because
     each was capped at its own 75th percentile, **exactly 25 percent of
     iterations were capped under every method by construction**. A method that
     understates the upper tail should conclude that capping buys less. It could
     not.

     The cap is now the 75th percentile of the VALUES a specifier holds,
     unweighted, applied to every method and to the true parent. It is what a
     practitioner can compute, and it exists on the empirical arm, where no
     parent does. **The share of iterations capped now runs from 0.277 under
     `Lognormal, Uniform` to 0.366 under `Normal, Uniform`**, which is the
     signal the old form destroyed.

     **WHAT IT MOVED.** Every `capecc_*` column of `TABLE_PLCAResults.csv`.
     `capecc_red_mean` -0.8623 to -0.8349 and `capecc_perc_mean` -0.1712 to
     -0.1698; the rank-frequency columns move much further and for a second
     reason, decision 116.

116. **2026-09-18, Stage 2e review. THE `(1-capecc)` DIVISOR HAD TO GO WITH THE
     CAP IT NORMALIZED. Stage 2g still owns the metric.** `[DELEGATED, 2e had
     no choice]` The roadmap gives this divisor to Stage 2g and this stage did
     not go looking for it: decision 115 made it wrong.

     It scaled a count over all iterations by 1 / 0.25, which was exact only
     because the old cap bound in exactly 25 percent of iterations for every
     material. Under an absolute cap the bound share is a property of the
     material and the method, so the divisor scaled by a number that is no
     longer the right one and a pLCA's four columns summed to 1.25.

     The frequency is now a plain count over the Monte Carlo draws: the share of
     iterations in which capping THIS material both bound and gave the largest
     reduction of the four. Two sums are then readable -- across the four ranks
     of one material, the share of iterations in which its own cap bound; across
     the four materials, the share in which any cap bound.

     **Filling untouched iterations with zero was tried and is worse**, because
     in the roughly quarter of iterations where no cap binds at all, four zeros
     tie for first and the tied average rank belongs to no integer rank, so
     those iterations vanish from every column instead of showing up as the
     shortfall. They stay excluded.

117. **2026-09-18, Stage 2e review. THE CAPPED DRAW IS EXACT, AND THE LAST
     REJECTION SAMPLER IN THE STUDY IS GONE.** `[DELEGATED, 2e chose]` A full
     run failed on it: a fitted model had so little mass below its material's
     cap that a bounded redraw loop could not bring the column below it.

     Redrawing until a value lands below the cap samples from the model
     conditioned on being below it, so `ppf(u * F(cap))` is the same
     distribution in one step. **Decision 50 had already settled that sampling
     in this study is by inverse CDF and never by rejection**; this loop was the
     last rejection sampler left in it. Tested against the loop it replaces, the
     two agree on every quantile from the 2nd to the 98th.

     It also makes the strategy paired across methods with ONE uniform block
     where the loop needed a cache of variates indexed by redraw pass, and it
     reports the case it cannot serve -- no mass below the cap -- instead of
     returning values still above a cap labeled as capped.

118. **2026-09-18, Stage 2e review. WHAT SWITCHING METHOD DOES TO THE DECISION A
     DESIGNER ACTUALLY MAKES: almost nothing. This is the stage's strongest
     result and it is a NULL.** `[AUTHOR]` The author's idea: treat swapping a
     material as the intervention, so two options share three materials and
     differ in the fourth. Implemented with the shared materials on the SAME
     random draws, which is dependent sampling, and with the replacement's use
     intensity carrying a controlled expected saving -- necessary because every
     dataset here is normalized to a mean of 1.0, so a substitution alone would
     change the expected total by nothing.

     800 option pairs, the truth being the same comparison run with the true
     parents. P(option B beats option A):

         B is claimed to save   0 pct   1 pct   2 pct   5 pct  10 pct  20 pct
         the truth              0.505   0.529   0.554   0.629   0.754   0.953
         spread over six UQ
           methods              0.006   0.006   0.007   0.012   0.020   0.012

     **The choice of UQ method changes the stated probability that a
     substitution is an improvement by at most two percentage points, and every
     method is within two and a half points of the truth.** The normal is
     consistently the most optimistic and the kernel estimate the least, which
     is the same ordering as everywhere else, but the gaps are small enough that
     no design decision turns on them.

     **Why this is the result to lead with rather than bury.** Every other
     comparison in this study is between a method and another method, or between
     a method and a target it was fitted to. This is the decision a designer
     makes, scored against the right answer, and the answer is that the choice
     is safe. The stage prompt anticipated exactly this -- "it is also the only
     experiment that can show the method differences do not matter at the
     decision level, and if that is what it shows, that is the cleanest result
     this paper could report" -- and asked for it either way.

119. **2026-09-18, Stage 2e review. WHERE THE CHOICE OF METHOD DOES MATTER: the
     upper tail, the compliance statement, and the value of a specification
     policy.** `[AUTHOR]` The same truth run, on the statements that are not a
     comparison.

     **Every method understates the building's 90th percentile**, by 0.09 to
     0.36 on a four-material building whose total averages 4.0, and they
     disagree about the compliance statement in both directions: at a budget the
     truth meets 90.0 percent of the time, `Lognormal, Variable` reports 91.3
     percent and `Normal, Uniform` reports 86.8.

     **The specification policy is where the normal fails hardest.** Against a
     true mean saving of **5.39 percent** of the building, the normal reports
     6.19 and the lognormal 4.88, while the KDE reports 5.22 to 5.42; asked for
     the chance of achieving at least a 5 percent building-level saving, the
     truth is **23.2 percent**, the normal says **30.5** -- an overstatement of
     7.4 points -- the KDE says 23.9 to 24.5 and the lognormal 22.9.

     **The quantity strategy is method-independent to four decimal places**, and
     the contrast is worth stating: using 25 percent less of a material is a
     deterministic fraction of its own contribution, so no distributional
     assumption enters, while specifying a cap acts entirely through the upper
     tail, which is exactly what the methods disagree about.

120. **2026-09-18, Stage 2e review. THE METHODS FAIL ON THE SAME MATERIALS, AND
     THE FAMILIES FAIL IN OPPOSITE DIRECTIONS.** `[AUTHOR]` Asked for directly:
     "it's important to know if the different UQ methods are failing in the same
     direction or not."

     **Direction.** On a material's estimated contribution the normal is biased
     HIGH (+0.044 uniform, +0.050 variable), the lognormal LOW (-0.038, -0.027)
     and the kernel estimate is nearly unbiased (-0.014, +0.003). On a
     material's 95th percentile they all fail the same way -- every one
     understates it, the KDE by 0.085 and the normal by 0.193.

     **Materials.** Per-material errors correlate **0.892 to 0.970 between
     methods that share a weighting scheme** and only **0.581 to 0.714 across
     weighting schemes**, and all six err in the same direction on **51.2
     percent** of materials against about 3 percent if they were independent. So the dominant axis of disagreement
     is the WEIGHTS and not the family, the three families make nearly the same
     error on the same material, and **choosing a different family does not
     hedge the risk**.

121. **2026-09-18, Stage 2e review. KNOWING MARKET SHARES BUYS 17 PERCENT;
     GUESSING THEM WITH A FLAT DIRICHLET CAPTURES A THIRD OF THAT ON THE
     MAGNITUDE AND NONE OF IT ON THE RANKING.** `[AUTHOR]` The
     oracle-weight counterfactual, framed as the author required: **the contrast
     is between knowing market shares and guessing them, not between two
     weighting schemes**, and nothing here says uniform weighting is better.

     1,200 pLCA groups against the market-weighted parent, mean absolute error
     in a material's estimated contribution:

         family      uniform   variable (flat Dirichlet)   oracle
         KDE          0.1285            0.1215             0.1064
         Lognormal    0.1243            0.1168             0.1007
         Normal       0.1620            0.1629             0.1550

     On the rank-1 frequency the oracle gains as much again -- KDE 0.0815
     uniform, 0.0818 variable, 0.0717 oracle -- **and the stand-in gains
     nothing at all**.

     **Read it as the cost of the stand-in.** Variable weighting is better than
     uniform on the magnitude, is a wash on the ranking, and would be better
     than both if the shares were known. What separates the oracle from the
     realized weights is noise this generator introduces by construction and the
     real world does not have, which is decision 79's finding reaching the
     decision level.

122. **2026-09-18, Stage 2e review. DISPERSION ENTERS THE SAFE-LEAD RULE AND
     CANNOT REPLACE IT, BECAUSE THE NATURAL WAY TO WRITE IT DOWN SATURATES.**
     `[AUTHOR]` The author asked twice, and the second time pointed out that the
     first answer had not addressed the question: "I figured this would be
     framed in terms of how many standard deviations apart they are or something
     like that." That is the right instrument and the first test did not use it.

     **WHAT WAS WRONG THE FIRST TIME.** The flip was regressed on log(ratio) and
     log(CV) as separate additive terms. The quantity the author named is the
     standardized separation, which for a leader whose mean contribution is `r`
     times the runner-up's is

         z = (r - 1) / sqrt((r * CV_lead) ** 2 + CV_second ** 2)

     and log(r) is the wrong numerator near r = 1, where every flip happens: the
     gap goes to zero much faster than log(r) does. The first test therefore
     understated dispersion by construction.

     **WHAT THE PROPER TEST SAYS.** Dispersion moves the risk at a fixed lead,
     and by a factor of two on solid counts: at a lead of 1.6 to 2.2 the flip
     rate runs **4.2 percent** for a pair worth 0.3 to 0.6 standard deviations
     (17,055 comparisons) and **2.0 percent** for one worth more than 1.6
     (7,845). At a lead of 2.2 to 3.5 it runs 2.6 percent to 0.1 percent.

     **AND THE RULE BARELY MOVES.** The lead needed for a 1 percent risk is
     **2.24** when both materials sit at a coefficient of variation of 0.25 and
     **2.36** at 1.5, across a six-fold range of dispersion. So the answer to
     decision 107 is unchanged and now rests on the right instrument.

     **WHY IT CANNOT BE STATED IN STANDARD DEVIATIONS AT ALL, which is the part
     worth printing.** The standardized separation SATURATES: as the lead grows,
     `z` tends to `1 / CV_lead`, because the leading material's own spread grows
     with its size. The median material in this corpus has a coefficient of
     variation of 0.55, so **it can never be more than about 1.8 standard
     deviations clear of a smaller material however large its lead** -- the
     observed median z goes 0.09, 0.76, 1.19, 1.60, 1.84 as the lead goes from
     1.2x to over 20x, pinned against its own ceiling of 1.79 to 1.86. A rule
     written in standard deviations could not tell a 10x lead from a 100x one.
     The ratio can, which is why it is the rule.

     `TABLE_PLCAFlipByLeadAndSpread.csv` is the table to print: the risk at a
     given lead, split by how many standard deviations that lead is worth, with
     counts beside every cell because the extreme corners are thin.

122b. **2026-09-18, Stage 2e review. A SMALL BIAS PER MATERIAL IS A LARGE ERROR
     FOR A BUILDING, because bias adds and noise does not.** `[AUTHOR]` Asked
     whether the bias directions of decision 120 were significant or minimal.
     They are minimal per material and decisive per building.

     On one material a method's bias is about a sixth of its noise: 0.044
     against 0.269 for `Normal, Uniform`. But summing four materials multiplies
     the bias by four and the noise by two, and the study's own numbers show the
     first exactly:

         method                bias per     building      building
                               material     bias          bias, pct
         Normal, Variable       +0.0498      +0.1992        +4.98
         Normal, Uniform        +0.0441      +0.1764        +4.41
         KDE, Variable          +0.0035      +0.0139        +0.35
         KDE, Uniform           -0.0141      -0.0564        -1.41
         Lognormal, Variable    -0.0273      -0.1093        -2.73
         Lognormal, Uniform     -0.0381      -0.1524        -3.81

     The implied and observed building columns agree to four decimal places
     because the bias is exactly additive. **So the choice of method shifts a
     whole building's estimate by up to nine percentage points from end to end,
     and running more materials will not average it away**, while the random
     part falls as one over the square root of the count. `TABLE_PLCABias.csv`.

123. **2026-09-18, Stage 2e review. Every negative tick label this project has
     ever drawn was a Unicode minus.** `[DELEGATED, 2e chose to fix]`
     `FIGURE_STYLE.md` requires plain ASCII and names the Unicode minus
     explicitly; nothing had ever set `axes.unicode_minus`, so matplotlib's
     default U+2212 went into every figure with a negative axis value. One line
     in `figstyle.apply`, and a test that draws a figure and asserts its tick
     labels are ASCII. It reaches the figures built since the style guide
     existed; the older ones do not call `figstyle` and are Stage 3's.

124. **2026-09-18. THE PEDIGREE MATRIX BELONGS IN STAGE 2h, AS A SWEEP OVER THE
     GEOMETRIC STANDARD DEVIATION AND NOT AS A CHOICE OF SCORES.** `[AUTHOR]`
     The author's observation, and it reframes what this paper contributes:
     "we're comparing data-driven methods. Existing probabilistic LCA methods
     like the pedigree matrix aren't data driven, they're driven by formulaic
     expert judgment in the absence of data... our contributions in terms of how
     far apart a probabilistic model can be until it makes a difference will be
     very important here."

     **WHY THE COMPARISON IS POSSIBLE AT ALL.** This study's yardstick -- a model
     this far from another changes the answer this often -- does not care how
     either model was built. So a judgment-driven model can be placed on the same
     axis as a data-driven one without any claim that the two approaches are
     comparable in kind, which is the claim that would not survive review.

     **WHAT TO BUILD.** A pedigree model here is a lognormal whose geometric MEAN
     is the single value a practitioner would report for the category and whose
     geometric STANDARD DEVIATION comes from the basic uncertainty plus the five
     indicators rather than from the data. It is one more entry in
     `src/families.py` and reuses everything else unchanged: the truth run, the
     interventions and the design comparison all take a fitted model and ask
     what it does.

     **SWEEP THE GSD; DO NOT CHOOSE SCORES.** Picking pedigree scores for a
     synthetic dataset is a judgment this project would then have to defend, and
     it is not defensible, because the scores describe a data COLLECTION context
     that a generated dataset does not have. Sweeping the geometric standard
     deviation across the range the matrix produces for plausible scores avoids
     it entirely and answers a better question: **at what GSD does a
     judgment-driven model start to give different answers from a data-driven
     one**, measured against the truth and against the calibrated thresholds this
     project already has.

     **The deliverable is one sentence of the form** "a pedigree model whose
     geometric standard deviation is within X of the data's own does not change
     the answer, and beyond that it does", which is the form a practitioner
     without the data can act on. Owner 2h.

125. **2026-09-19, Stage 2f. THE TWO NORMALITY COLUMNS WERE TWO DIFFERENT
     STATISTICS. Both are now SHAPIRO-FRANCIA, and the columns are renamed
     `fit_norm_SF` and `fit_lognorm_SF`.** `[AUTHOR]` "Decided: use
     Shapiro-Francia for both columns."

     `customstats.shapiro_wilk_weighted` returned scipy's true Shapiro-WILK W
     when the weights were uniform and a Shapiro-FRANCIA W' when they were not.
     So `fit_norm_SW` and `fit_norm_SW_uw` were not one statistic under two
     weightings, and **four panels of the main characteristic figure compared
     them as though they were**. Discrepancy entry 11 had this open since
     Stage 0.

     **Why Shapiro-Francia and not Shapiro-Wilk.** It is the only one of the two
     with a weighted form. Shapiro-Wilk's coefficients come from the covariance
     matrix of normal order statistics at a given n and there is no accepted
     weighted generalization, so choosing it would mean either dropping the
     variable-weighted column or inventing one. Shapiro-Francia is the squared
     correlation between the order statistics and their normal scores, which
     takes weights directly, and it is the choice that makes the comparison the
     panels claim to show.

     **The columns are RENAMED, not silently redefined.** A column called
     `fit_norm_SW` holding a Shapiro-Francia statistic is the kind of thing this
     project keeps having to catch. `shapiro_wilk_scipy` is kept as the true
     Shapiro-Wilk for `audits/shapiro_estimator.py` and is called from nothing
     in the production path.

     **WHAT MOVED, and it is confined exactly where it should be.** Only the two
     UNIFORM columns; the variable-weighted ones were already Shapiro-Francia
     and are BIT-IDENTICAL on both arms. Median absolute change in
     `fit_norm_SF_uw`, empirical arm by stratum: **0.0103** at n = 3-9,
     **0.0062** at n = 10-99, **0.0029** at n = 100-999, **0.00005** above
     n = 1,000. Arm mean 0.7862 to 0.7834. Largest single change on either arm
     0.0358.

     **THE OLD DOCSTRING'S EQUIVALENCE CLAIM FAILS WHERE THE STUDY NEEDS IT.**
     It said the two are indistinguishable for n >= 20. Measured: median
     absolute difference 0.006 at n = 20 and 0.0004 at n = 2,000, so that is
     about right -- and the smallest stratum here is n = 3 to 9, where it is
     0.010 with a maximum of 0.029. At n = 3 the two are identical.

     **NOTHING ELSE MOVED, AND THAT IS CHECKED RATHER THAN ASSUMED.** No fit, no
     W1 score and no pLCA number can move, because the Shapiro statistic is a
     reported characteristic and enters nothing. The GENERATOR CALIBRATION is
     bit-identical: `coverage.distribution_comparison` reads only
     variable-weighted columns, so all ten standardized distances agree to the
     last digit and **no generation decision is reopened**. The corpus was
     recomputed by `corpus.remetric_corpus`, not regenerated --
     `corpus_2026-09-19`, with `values.parquet`, `parents.json.gz`,
     `combos.csv`, `parents_spec.json.gz` and `mode_labels.parquet` all verified
     byte-identical to `corpus_2026-09-15b`. Decisions 47, 48, 55 and 58 stand.
     Discrepancy entry 114.

126. **2026-09-19, Stage 2f. `_royston_pvalue` WAS WRONG IN TWO WAYS, not the
     one that was known, and is now correct to 4e-12 against scipy.**
     `[AUTHOR]` "leaving a known-wrong p-value in a public deposit is not
     acceptable."

     1. The `4 <= n <= 11` branch applied the `n >= 12` polynomials, which are
        in log(n), to a range whose Royston coefficients are polynomials in
        **n itself**, and subtracted the gamma shift from the transformed
        variable instead of applying Royston's `-log(gamma - log(1 - W))`
        re-expression. **At n = 10 and 11 it returned 1.0000 where the correct
        value is about 0.50**, and the largest observed error was 0.99.
     2. **The `n >= 12` branch was also wrong, which nobody had noticed.** It
        evaluated the sigma polynomial at `log(log(n))` where Royston evaluates
        it at `log(n)`, so the p-value was wrong at EVERY sample size, by up to
        0.077 at n = 5,000.

     **Nothing reported ever depended on it**: only the statistic is kept, by
     decision, because a p-value at n = 77,548 measures the sample size rather
     than the departure from normality. It is fixed because the repository is a
     public deposit.

     The statistic the module now returns is Shapiro-Francia, so it gets
     **Royston's (1993) W' transform** rather than the Shapiro-Wilk one, and
     that transform returns NaN outside 5 <= n_eff <= 5000 instead of
     extrapolating a fit past the range it was made on. Discrepancy entry 115.

127. **2026-09-19, Stage 2f. THE PANEL COUNT IS SETTLED: the manuscript's 18 is
     wrong, 19 was right for the figure it describes, and the code now draws
     21.** `[DELEGATED, 2f confirmed]`

     The figure draws every characteristic both arms carry except `mean_uw`,
     which is identically 1.0 by construction because every dataset is divided
     by its own unweighted mean (decision 6). Before Stage 2a that is **19**:
     **eight characteristics with a uniform and a variable version** --
     coefficient of variation, entropy, the normal and lognormal Shapiro
     statistics, kurtosis, the modality index, skewness, weight of outliers --
     giving 16 panels, plus **three single panels**: dataset size, the
     uniform-to-variable Wasserstein distance, and **the variable-weighted
     MEAN**, which is the panel the earlier count could not name.

     Stage 2a added Silverman's critical bandwidth under both weightings
     (decision 23), so the figure as the code now draws it has **21**.

     **The number the manuscript should print is neither**, because this stage's
     whole purpose is that 21 marginal panels represent about four to five
     independent quantities. The full candidate count belongs in the supplement
     and the survivor count in the main text. Discrepancy entry 116.

128. **2026-09-19, Stage 2f. The visible-mode counts were a 500-dataset sample
     and are now the whole corpus, because the sample cost more than it saved.**
     `[DELEGATED, 2f chose]` Notebook 1 drew 500 of 10,000 synthetic datasets
     for `TABLE_VisibleModes.csv`. Timed: all 10,000 take **under a minute**.

     The sample was what kept `modes_fitted` and `modes_scipy_default` out of
     this stage's reduction as first-class predictors -- a complete-case model
     over 402 usable rows of 10,000 drops the corpus -- and it would have made
     the complete-case cost this stage was asked to report a statement about the
     sampling rather than about the undefined kurtosis it is about.

     **What moves**: the synthetic share with one, two and three or more visible
     modes, by sampling error only, the 500 having been a random draw. The
     empirical arm's method is unchanged and now covers every category with
     n >= 8. Any figure or sentence quoting a synthetic mode share must be taken
     from the rebuilt table and not from decision 82's text.
     Discrepancy entry 117.

129. **2026-09-19, Stage 2f. THE TWENTY-ONE-PANEL CHARACTERISTIC FIGURE IS WORTH
     FIVE PANELS, AND THE FIVE ARE TWO QUANTITIES: DISPERSION AND DATASET
     SIZE.** `[AUTHOR]`

     **NARROWED BY DECISION 135: this holds for the LEVEL of a method's score
     and is NOT the answer to which method to use.** On the choice between two
     families, with size and dispersion already in the model, five further
     characteristics add 0.10 to 0.14 of explained variance on the real arm.
     Read 135 before quoting the five-panel figure as the reduction's result. The stage's deliverable, and the author said in advance
     that a reduction landing on size and little else is a result rather than a
     failure. It lands on size AND dispersion, which is slightly richer.

     Pooled over both target families, both arms, all six methods and both model
     families -- 96 models, ranked by mean permutation-importance rank on the
     held-out fold, with the definitional candidate of decision 130 removed:

         metric         mean rank   in the top five
         coeffvar          3.09         84 pct
         coeffvar_uw       3.80         78 pct
         entropy           3.93         83 pct
         n                 5.06         70 pct
         entropy_uw        5.24         70 pct
         mean              6.79         49 pct
         ...
         crit_bw_1        13.2           0 pct
         modes_fitted     19.5           0 pct

     **THE FIVE COLLAPSE TO TWO.** `entropy` is a histogram entropy over 256
     bins, so it counts how many bins the data fill: its spline R2 on log(n)
     alone is **0.924 on the corpus and 0.839 on the real arm**, at Spearman
     +0.88 and +0.90. It IS dataset size. And `coeffvar_uw` is `coeffvar`, at a
     correlation of 0.979. What is left is dispersion and size, which are nearly
     independent of each other -- +0.11 empirical and +0.30 synthetic -- which is
     why neither looked like the whole answer alone. That orthogonality is the
     same one decision 96 found for the weighting question, arrived at from a
     different direction.

     **THE EFFECTIVE DIMENSION SAYS THE SAME THING BEFORE ANY MODEL IS FITTED.**
     The participation ratio of the correlation eigenvalues over all 23
     candidates is **4.33 on the empirical arm and 5.43 on the synthetic**, so
     twenty-three marginal panels were always showing about four or five
     independent quantities.

     **THE SET IS STABLE AND THE ORDER IS NOT.** The audit script and the
     notebook run on different random streams and return the same five; the mean
     ranks agree to 0.03 to 0.25 and the second and third places swap. The paper
     states a SET, never a ranking within it. `TABLE_ReductionSurvivors.csv`.

130. **2026-09-19, Stage 2f. `w_v_uw_wasserstein` IS THE DEFINITIONAL TERM OF A
     FIT SCORE, NOT A PREDICTOR OF IT, and every survivor ranking is reported
     with and without it.** `[DELEGATED, 2f measured]`

     Every model in this study is scored against the VARIABLE-weighted empirical
     CDF, including the three uniform-weighted fits, so a uniform-weighted model
     is charged a distance no estimator can remove. That distance is exactly the
     Wasserstein distance between the uniform-weighted and variable-weighted
     versions of the dataset, which is what `w_v_uw_wasserstein` measures.

     **Measured, its Spearman correlation with `w1_definitional` is 1.000000 for
     all three uniform-weighted methods on both arms.** That is an identity, not
     a relationship, and on its strength alone it reaches a Spearman of **0.966**
     with the in-sample W1 of `KDE, Uniform`. A reduction that ranked it first on
     a fit target would be rediscovering an identity.

     **ITS STANDING ON THE DOWNSTREAM ERROR IS REAL, and that is the part worth
     keeping.** The variable-weighted methods have a definitional term of exactly
     zero, and it still correlates **0.72, 0.72 and 0.57** with the error in a
     material's estimated contribution for the three uniform methods and 0.56 to
     0.59 for the variable ones. So it predicts how wrong the ANSWER is without
     any identity to lean on.

     `reduction.definitional_check` reports the correlation per method and
     candidate and flags an exact identity; `DEFINITIONAL_CANDIDATES` names it.
     **A later stage adding a candidate derived from the scoring target must add
     it to that list.** Discrepancy entry 118.

131. **2026-09-19, Stage 2f. THE TWO TARGETS DISAGREE ABOUT DATASET SIZE, AND THE
     GOODNESS-OF-FIT FIGURE WAS WEIGHTING THE WRONG THINGS.** `[AUTHOR]`

     **THERE ARE THREE TARGETS, NOT TWO. Decision 135 adds the one the paper
     actually asks about**, the choice between two families, and the ordering
     under it differs from both of the two here. This entry stands on its own
     terms; it is not the whole comparison.

     This is what the stage's instruction to run the reduction twice was for: a
     characteristic that predicts the distance between a fitted curve and its
     target, but not the error in the ANSWER, is not worth keeping.

     Rank against the FIT score beside rank against the DOWNSTREAM error, where a
     NEGATIVE shift means the characteristic matters more for the answer:

         n                    6.90 -> 4.42   -2.48
         skewness_uw         15.02 -> 12.65  -2.38
         kurtosis_uw         14.35 -> 12.04  -2.31
         entropy_uw           7.13 ->  4.90  -2.23
         ...
         coeffvar_uw          3.38 ->  5.40  +2.02
         fit_lognorm_SF      14.13 -> 16.35  +2.23
         fit_norm_SF          9.88 -> 13.85  +3.98
         weight_outliers     13.25 -> 17.29  +4.04

     **Size and its proxy move UP when the target becomes the answer; the shape
     statistics move DOWN.** `n` has the largest negative shift of any
     characteristic and `weight_outliers` and the normal Shapiro statistic the
     largest positive ones. So the figure this stage replaces was over-weighting
     how well a curve fits relative to what decides the probabilistic LCA's
     error, and the correction points at dataset size -- which is the mechanism
     decisions 84, 86 and 88 already identified from the method-choice side.

     **ONE OUTPUT IS ALMOST UNPREDICTABLE FROM THE DATA, AND IT IS THE RANKING
     ONE.** The models reach an out-of-sample R2 of 0.62 to 0.66 for the error in
     a material's estimated contribution, 0.52 to 0.54 for its 95th percentile
     and 0.54 for its share of total variance -- and **0.085 to 0.096 for the
     error in its chance of leading**. That is not a failure of the model: a
     rank-1 frequency is a property of the GROUP of four materials, not of the
     dataset, so the dataset's own characteristics cannot carry it. It is a third
     independent argument for decision 102's demotion of the ranking metrics.

132. **2026-09-19, Stage 2f. THE THREE MODALITY MEASURES ARE NOT THREE READINGS
     OF ONE THING, THE ONE THAT PREDICTS IS SILVERMAN'S CRITICAL BANDWIDTH, AND
     IT PREDICTS NOTHING OF ITS OWN.** `[AUTHOR]` The question Stage 2a-2 could
     not answer, and the answer reproduces this stage's own lesson.

     **SUPERSEDED IN ITS CONCLUSION BY DECISION 134, WHICH IS THE ONE TO READ.**
     This entry tested three modality measures and none of them was the
     author's own index at the bandwidth the study fits -- the only combination
     that had never been computed. At that bandwidth the author's index is the
     SECOND-best predictor of which method to use, of 23 candidates, at
     p = 0.0002. What survives from this entry is that the mode COUNTS carry
     nothing and that the measures disagree with each other; the claim that no
     modality measure has predictive content of its own is withdrawn.

     **They disagree, which had to be established first.** Spearman between
     `modality_index` and `modes_fitted` is **+0.168 on the real arm and +0.018
     on the corpus**; between `crit_bw_1` and `modes_fitted`, **+0.283 and
     -0.115**. A negative correlation between two measures of the same property
     settles that they are not measuring the same property.

     **Offered ALONE over a spline in log(n), Silverman's critical bandwidth wins
     by a distance**: mean incremental R2 **0.210 empirical and 0.175 synthetic**,
     significant on every one of the 12 and 18 models. The continuous index adds
     0.128 and 0.042. **The mode COUNTS add almost nothing** -- `modes_fitted`
     0.012 on both arms, and on the empirical arm it is not significant on ANY of
     the twelve models, median p = 0.27.

     **AND IN THE FULL MODEL THE CRITICAL BANDWIDTH IS 14th OF 22 AND IN THE TOP
     FIVE OF ZERO OF 96 MODELS.** It correlates **+0.54 empirical and +0.60
     synthetic with the coefficient of variation**, so everything it appeared to
     carry over size alone is dispersion it travels with. The two mode counts
     rank 21st and 22nd with a mean importance of essentially zero.

     **So the honest statement is one sentence with two halves:** among the
     modality measures, only Silverman's critical bandwidth carries any signal,
     and none of that signal is its own. This CONFIRMS decision 88, which found
     multimodality last of eleven, and explains it: decision 88 tested a mode
     COUNT, which is indeed worthless, and the critical bandwidth is not
     worthless -- it is dispersion under another name.

     **It is also this stage's own trap, appearing inside its own results**, and
     that is why the paper should show it: tested against size alone a
     characteristic can look decisive and contribute nothing once the rest of the
     set is present. Discrepancy entry 119.

133. **2026-09-19, Stage 2f. THE CORPUS HAS MARGIN WHERE IT DOES NOT MATTER AND
     NONE WHERE IT DOES, SO THE GENERALIZATION CLAIM IS NARROWER THAN THE
     COVERAGE FIGURE SUGGESTS.** `[AUTHOR]` The conceptual question the stage was
     given: the tuning objective matches the SHAPE of the synthetic
     characteristic distribution to the empirical one, while the study also needs
     to SPAN that space with margin so its conclusions generalize past the
     categories EC3 happens to hold, and those two goals can pull apart. They do.

     `margin_above` is how far the synthetic range reaches past the empirical
     maximum, in units of the empirical range. Negative means the corpus does not
     reach the empirical maximum at all.

         the five survivors        margin above the empirical max
           coeffvar                    -0.629
           coeffvar_uw                 -0.656
           n                           -0.678
           entropy                     +0.126
           entropy_uw                  +0.132

         characteristics that predict nothing
           modality_index              +6.864
           kurtosis                    +5.249
           skewness                    +7.186

     **Median margin: -0.629 for the five survivors and +0.509 for the other
     seventeen.** Spearman between importance rank and margin is **+0.484**, and
     since a lower rank means more important, positive means the metrics that
     matter have the least headroom.

     **WHAT THIS DOES AND DOES NOT SAY.** It does not say the corpus fails to
     cover the empirical data: 98 to 100 percent of real datasets sit inside the
     synthetic range on every metric, so INTERPOLATION is supported. It says that
     EXTRAPOLATION beyond the range EC3 happens to contain is supported on the
     characteristics that carry no signal and not on the three that do.

     **It sharpens decision 48 rather than reversing it.** That decision accepted
     the dispersion shortfall as a stated limitation on the grounds that what the
     corpus cannot reach is the shape of a contaminated EC3 category rather than
     the shape of a material -- an argument about WHICH datasets are uncovered.
     This adds that the shortfall sits on the single most predictive
     characteristic in the study, which bounds how far the conclusions carry
     regardless of which categories are uncovered. **Generation stays closed**,
     decisions 47, 48 and 55 unchanged; what the manuscript owes is the
     limitation stated in these terms. Discrepancy entry 120.

134. **2026-09-21, Stage 2f review. THE AUTHOR'S MODALITY INDEX WAS NEVER THE
     PROBLEM. IT WAS ON THE WRONG BANDWIDTH, AND AT THE RIGHT ONE IT IS THE
     SECOND-BEST PREDICTOR OF WHICH METHOD TO USE. This SUPERSEDES decision
     132 and REVERSES decision 23.** `[AUTHOR]` "I had a previous method I was
     pretty happy with, and you got rid of it in favor of methods that
     apparently don't work."

     The author's measure is `customstats.estimate_maxima`: the summed heights
     of the local maxima of a kernel density, less the summed heights of the
     local minima, over the tallest peak. Subtracting the minima is what stops
     two maxima with a shallow dip between them registering as two full modes.

     **It was hardcoded to Scott's rule.** It was written when Scott was the
     study's bandwidth; decision 54 moved the study to `silverman_guarded` in
     Stage 2b and nothing brought this measure with it. Four stages then
     measured the author's idea at a bandwidth the study had abandoned.

     **WHAT THE BANDWIDTH WAS WORTH, on the real categories, predicting which
     of the kernel estimate and the three-parameter lognormal fits better, with
     dataset size AND dispersion already in the model, 127 datasets, Bonferroni
     threshold 0.0022 for the 23 candidates tested:**

         measure                            incremental R2    p     rank of 23
         the author's index, FITTED bw          0.1114     0.0002        2
         the author's index, Scott's bw         0.0179     0.6944       21
         Silverman's critical bandwidth      0.0777-0.1070 0.0003-0.007 3, 15
         visible mode COUNT, fitted bw          0.0100     0.3340       23
         visible mode COUNT, Scott's bw         0.0155     0.1817       22

     **The same measure goes from 21st of 23 and entirely insignificant to 2nd
     of 23 at p = 0.0002. Nothing about it changed but the smoothing.**

     **DECISION 23 IS REVERSED.** It discarded this metric because it "spans
     only 1.000 to 1.159 across the 138 empirical datasets, so as a count it is
     constant at 1". That range is a property of Scott's oversmoothing: at the
     bandwidth the study fits, the same index spans **1.000 to 1.249** on the
     real arm. The readout was not the defect; the bandwidth was.

     **AND THE REPLACEMENT WAS WORSE THAN WHAT IT REPLACED.** The visible mode
     COUNT that Stage 2a-2 adopted in its place (decision 38) is the worst of
     all 23 candidates on this target at either bandwidth. Counting modes
     throws away exactly the information subtracting the minima preserves.

     `estimate_maxima` now takes a `bw_method`, and `modality_index_fitted` is
     computed beside the existing column, which is untouched because it is what
     every earlier stage quoted. **`modality_index_fitted` is the modality
     measure the paper should report.** The corpus was recomputed for the new
     column, not regenerated: values, parents, groupings and mode labels are
     byte-identical and no existing characteristic moved. Discrepancy entry 123.

135. **2026-09-21, Stage 2f review. THE REDUCTION ANSWERED THE WRONG QUESTION.
     Ranking characteristics by how well they predict the LEVEL of a score is
     not an answer to "which method should I use". This NARROWS decisions 129,
     131 and 132.** `[AUTHOR]` Raised by the author on skewness: "I'm surprised
     skew doesn't matter. Wouldn't left skew make it difficult for lognormal or
     normal to get a good fit?"

     The instinct was right and the reduction could not see it. It modeled the
     LEVEL of each method's own W1, and the level is dominated by dispersion and
     dataset size because **every** method gets worse on spread data and on
     small samples. The paper asks which method to USE, which is the DIFFERENCE
     between two of them -- and a difference is about whose shape assumption
     fits, so it is exactly where skewness, kurtosis, lognormality, modality and
     outlier weight live.

     **On the real categories, predicting `log(W1_KDE / W1_lognormal)` within a
     weighting scheme, with size AND dispersion already in the model:**

         uniform weights                 variable weights
         mean                  0.125     lognormal Shapiro     0.140
         modality index (fit)  0.111     lognormal Shapiro uw  0.103
         critical bandwidth uw 0.107     critical bandwidth    0.091
         weight of outliers    0.105     critical bandwidth uw 0.078
         kurtosis              0.103     skewness              0.077
         kurtosis uw           0.100     modality index (fit)  0.075

     Base R2 0.483 and 0.513. The first five under uniform weighting and the
     first three under variable survive a Bonferroni correction for 23 tests;
     skewness at p = 0.0041 sits just past it, which is **real but not the
     strongest**, and the paper should say so rather than claim more.

     **TWO PROPERTIES OF THE RATIO MATTER AND ARE WHY IT IS THE RIGHT TARGET.**
     Taken within a weighting scheme it CANCELS the part of the score no
     estimator can remove, so `w_v_uw_wasserstein` becomes a legitimate
     predictor here where on the level it was the identity decision 130
     records. And it is scale free, so it does not inherit the level's
     dependence on how spread the data happen to be.

     **LEFT SKEW, which the author asked about specifically.** On the corpus,
     where 1,674 of 10,000 datasets are left skewed, they are harder for all
     six methods -- mean score 0.24 to 0.29 against 0.14 to 0.23 on the
     right-skewed ones -- so the author's expectation holds there. **The real
     arm has only 8 left-skewed categories of 147**, so it cannot support a
     claim either way, and that asymmetry is itself worth a sentence: real ECC
     data is almost never left skewed, and the corpus is the only place the
     question can be asked.

     **THE CORPUS IS THE WEAKER ARM FOR THIS WHOLE QUESTION.** Every increment
     above is an order of magnitude smaller on the synthetic arm, 0.01 to 0.03
     against 0.08 to 0.14, which is the coverage shortfall of decision 133
     showing up as a loss of statistical power rather than as a bias. Decision
     136 is the author's call on that.

     **The level and the answer targets are kept and reported**, because the
     contrast between the three is the finding; what changes is which one is
     presented as the answer. Discrepancy entry 124.

136. **2026-09-21, Stage 2f review. THE CHOICE ANALYSIS MOVES TO THE SYNTHETIC
     ARM, BECAUSE 127 REAL CATEGORIES CANNOT MEASURE IT AT ALL. Decision 134's
     "2nd of 23" is WITHDRAWN as an out-of-sample claim and decision 135's
     increments are withdrawn with it.** `[AUTHOR]` "Why not the synthetic
     datasets?? I know 127 categories isn't very many, that's why we created
     synthetic datasets. Why would we base our findings on just the empirical
     datasets? That would be an extremely flimsy approach to this analysis."
     And, on the significance criterion: "P-value is such an antiquated,
     arbitrary, silly metric. We can do way better than that."

     **WHAT WAS WRONG.** Decisions 134 and 135 ranked characteristics by
     IN-SAMPLE incremental R2 on 127 datasets. Adding a five-knot spline to 127
     points raises in-sample R2 by about **0.043 under the null**, so increments
     of 0.10 to 0.14 are part signal and part free lunch, and there was no way
     to tell which from the numbers reported.

     **THE SAME DATA, MEASURED OUT OF SAMPLE, AND THE REAL ARM'S R2 IS NOT A
     NUMBER.** An earlier draft of this entry quoted a base R2 of -0.724 for
     the empirical arm and the notebook's own run returned -2.441 on the same
     data through the same code. Neither is wrong and neither is a
     measurement: the only difference is which categories land in which fold.
     Twenty fold assignments, base model of size and dispersion, nothing else
     changed:

         arm         weighting   datasets   base R2 median   range         sd
         synthetic   Uniform       10,000       +0.280       +0.279/+0.280  0.000
         synthetic   Variable      10,000       +0.524       +0.523/+0.525  0.000
         empirical   Uniform          127       +0.326       -0.442/+0.370  0.169
         empirical   Variable         127       -0.952       -2.605/+0.337  0.900

     **The corpus reproduces its own R2 to three decimals on every reshuffle;
     the real arm does not reproduce its own SIGN.** A base R2 below zero means
     size and dispersion together predict the choice worse than predicting its
     mean. So the empirical arm cannot support this model, no ranking taken
     from it is a measurement, and quoting any single value from it -- a
     flattering one included -- is quoting one draw from a three-point-wide
     distribution. `TABLE_ReductionBaseStability.csv`.

     **THE SYNTHETIC ARM CAN, AND IT SAYS SOMETHING DIFFERENT.** 10,000
     datasets, out-of-sample gain over size and dispersion, fold spread beside:

         uniform-to-variable W1      uniform  +0.170 (sd 0.027)
         the variable-weighted mean  variable +0.034 (sd 0.007)
         weight of outliers          variable +0.026 (sd 0.009)
         visible mode COUNT, fitted  variable +0.019 (sd 0.010)
         kurtosis                    variable +0.018 (sd 0.012)
         the author's modality INDEX, fitted bandwidth
                                     uniform  +0.0015 (sd 0.007)
                                     variable +0.0027 (sd 0.011)
         Silverman critical bandwidth         +0.005 / +0.008

     **SO DECISION 134 IS NARROWED, NOT REVERSED.** The bandwidth defect it
     found is real and the fix stands: `estimate_maxima` was hardcoded to
     Scott's rule after the study moved to a guarded Silverman, and
     `modality_index_fitted` is the corrected column. What does NOT survive is
     the claim built on it. Out of sample the index is indistinguishable from
     zero, and the visible mode COUNT -- which decisions 132 and 134 both called
     worthless -- is the modality measure forward selection actually keeps.
     **Decision 23's reversal stands on the bandwidth argument alone.**

     **SIGNIFICANCE IS NOW A CROSS-VALIDATED GAIN AGAINST ITS OWN FOLD SPREAD,
     and no p-value appears in any figure, table or claim of this stage.**

     **AND REDUNDANCY IS HANDLED BY SELECTION RATHER THAN BY CORRELATION
     PRUNING.** One-at-a-time increments credit skewness, kurtosis and the
     normality statistic separately for the same shape. Forward selection adds
     only what still helps once everything already chosen is in:

         uniform weights   uniform-to-variable W1, mode count, dispersion (uw),
                           entropy, outlier weight x2      R2 0.280 -> 0.581
         variable weights  the mean, outlier weight, mode count, dispersion
                           (uw), normal fit (uw), critical bandwidth
                                                           R2 0.525 -> 0.634

     **UNDER MARKET-SHARE WEIGHTS MANY EFFECTS ARE MEASURABLE AND ALL OF THEM
     ARE SMALL, which is a different claim from "only one matters".** 17 of 23
     candidate gains exceed their own fold spread, because that spread is only
     0.0028 on 10,000 datasets -- but the largest is +0.034 against a base of
     0.525. Under equal weights only 1 of 23 does, and it is the
     uniform-to-variable distance at +0.170. The paper should say the effects
     are real and small rather than absent, and should not use the count of
     things that clear a noise floor as a measure of how much they matter.

     The empirical arm is reported as a consistency check with its interval
     shown and is stated to be too small to confirm anything. No claim about
     which method to use rests on it. Discrepancy entry 126.

137. **2026-09-21, Stage 2f review. THE SIZE CAP STAYS AT 9,999, and the author
     was right to challenge it.** `[AUTHOR]` "Do you really think there's a
     material difference between a dataset of 10,000 and 30,000? I'm not saying
     explicitly one way or the other, but I want to challenge whether that's
     entirely necessary."

     There is not, on three measurements of `log(W1_KDE / W1_lognormal)`:

     **The curve is not accelerating.** Slope per decade of n on the corpus
     under uniform weighting: **-0.131** at n = 10-99, **-0.307** at 100-999,
     **-0.233** at 1000-9999. Extending to 31,025 is 0.49 of a decade and worth
     about -0.11, which moves no threshold.

     **The decision is already unanimous up there.** The kernel estimate is
     closer on **85.9 percent** of corpus datasets at n = 4000-9999 under
     uniform weighting and **96.2 percent** under variable. More size would
     refine a magnitude in a regime where the answer does not change.

     **The three real categories above the cap behave like the band below it.**
     `ReadyMix [5000-5999 psi]` at n = 14,366, `[3000-3999]` at 20,814 and
     `[4000-4999]` at 31,025 give log ratios of -0.351, -0.056 and -0.080 under
     uniform weighting, inside the spread of the 1000-9999 band rather than
     beyond it. They are 3 of 147 categories, 2.0 percent of the arm.

     Decision 14's reasoning is unchanged and this adds the measurement it
     lacked. The `margin_above` shortfall on `n` reported in decision 133 is
     therefore a number without a consequence.

138. **2026-09-21, Stage 2f review. THE DISPERSION GAP IS ONE CONTAMINATED
     CATEGORY, WIDENING THE GENERATOR COSTS THE PAPER'S CENTRAL QUANTITY, AND
     THE RECOMMENDATION IS NOT TO. THIS CONTRADICTS AN AUTHOR INSTRUCTION AND
     IS BROUGHT BACK RATHER THAN ACTED ON.** `[AUTHOR ASKED, MEASUREMENT
     DISAGREES, AWAITING THE AUTHOR]` The instruction was "Definitely widen
     dispersion", given against a stated gap of 2.58 against 6.93. That framing
     was a ratio of maxima, and the maxima are set by one category.

     **WHERE THE DISPERSION IS LOST, which no earlier audit had separated.** A
     dataset's coefficient of variation passes through a drawn TARGET, a solved
     PARENT and a finite SAMPLE. Measured at 400 datasets per configuration:

         CV target drawn    median 1.26   p99 13.93
         parent solved      median 0.70   p99  1.28   max 1.31
         sample drawn       median 0.54   p99  1.28   max 2.02

     **60 percent of targets come back `clipped_max_cv`**: the generator asks
     for a spread it cannot build. The binding constraint is the truncation,
     `Q1 / r**mult` and `Q3 * r**mult` with `r = q3/q1` capped at
     `1 + 1/min_q1_over_iqr` = 3. **`audits/dispersion_reach.py` swept the
     target and the floor and never swept `trunc_iqr_mult`**, which is the
     lever that actually binds, so decision 48's "not reachable by any
     parameter" was measured with that parameter held fixed.

     **IT IS REACHABLE, AND THE PRICE IS THE QUANTITY THE PAPER IS ABOUT.**
     Eight configurations at 440 datasets, full calibration objective; the
     seed-to-seed standard deviation of the objective is 0.0066:

         configuration                     objective  coeffvar  weighting  max CV
         current                             0.2251     0.380     0.275     1.65
         mult 5, floor 0.02, center +0.3     0.2666     0.188     0.661     3.52
         mult 8, floor 0.02, center +0.3     0.2638     0.262     0.355     1.83
         mult 8, floor 0.01, center +0.5     0.3063     0.297     0.869     2.01
         mult 12, floor 0.01, center +0.5    0.3917     0.325     1.168     1.92

     The best dispersion match halves the coefficient-of-variation distance,
     0.380 to 0.188, and **multiplies the uniform-to-variable Wasserstein
     distance by 2.4**, 0.275 to 0.661, while Silverman's critical bandwidth
     doubles and skewness goes 0.156 to 0.272. The overall objective worsens by
     **6.3 seed standard deviations**. This is the failure decision 39 already
     records: a configuration that improves the statistic being watched while
     the corpus gets worse on the quantity the study is built on.

     **AND IT STILL DOES NOT REACH THE TARGET.** The best candidate puts 0.2
     percent of datasets above a coefficient of variation of 2 against the real
     arm's 4.1 percent, and reaches 3.52 against 6.93.

     **THE GAP IS ONE CATEGORY, BY NAME.** Exactly **1 of 147** real categories
     sits above the corpus maximum of 2.576: **`Aggregates`, n = 378,
     coefficient of variation 6.929, skewness 12.5, kurtosis 163.9**. Six
     categories exceed 2.0 and one exceeds 2.5. `Aggregates` is the category
     decisions 48, 49, 60 and 61 all name as a contaminated EC3 bin -- the one
     where a relative outlier filter puts its upper bound at 41,238,610 times
     the median and trims nothing.

     **AND NOTHING DEPENDS ON IT, WHICH IS THE MEASUREMENT THAT DECIDES THIS.**
     Removing every category above the corpus maximum:

         kernel estimate closest, uniform    40.2 pct -> 40.5 pct
         kernel estimate closest, variable   34.6 pct -> 34.9 pct
         size crossover, uniform             n = 124.2 -> 122.1
         size crossover, variable            n = 204.0 -> 196.3

     **DECISION 133 IS NARROWED and its framing withdrawn.** It reported that
     the corpus has margin on the characteristics that predict nothing and none
     on the three that predict, and concluded that extrapolation is unsupported
     where it matters. The margin figures are correct; the conclusion is not,
     because the shortfall is a single contaminated category and excising it
     moves every headline by less than half a percentage point.

     **WHAT IS RECOMMENDED INSTEAD.** State the limitation in terms of what is
     actually missing: the corpus spans the dispersion of every real material
     category and does not span one contaminated EC3 bin, which is a statement
     about EC3's taxonomy rather than about the generalization of the result.
     **Generation stays closed unless the author overrules this**, decisions
     47, 48 and 55 unchanged. Discrepancy entry 127.

139. **2026-09-21, Stage 2f review. THE PRACTITIONER RULE, MEASURED: use a
     kernel estimate above 75 to 100 declarations, with market-share weights if
     you have them, and a three-parameter lognormal below. The cutoff is a
     basin, not a point.** `[AUTHOR]` "I want to find the most useful versions
     of those kinds of statements. Like 'KDE is best for n>100'. Is the cutoff
     actually right at 100 or is it elsewhere?"

     **THE CUTOFF IS 81 AND EVERYTHING FROM 68 TO 97 IS INDISTINGUISHABLE FROM
     IT. Decision 142 supersedes the "59 to 134" this entry first reported**,
     which came from a tolerance chosen rather than measured. Cost of the
     policy "kernel estimate with market-share weights above the threshold,
     three-parameter lognormal with equal weights below" against the
     unreachable per-dataset oracle, on 10,000 synthetic datasets scored
     against the market-weighted parent:

         always the kernel estimate          53.6 pct over the oracle
         n >= 81                             38.3   <- lowest
         always the lognormal               178.5

     `TABLE_ReductionPolicyCurve.csv` and `TABLE_ReductionThreshold.csv`.

     **IS ONE METHOD BEST NO MATTER WHAT? No, and the answer changes twice.**
     Share of datasets on which each method is closest to the truth:

         n            KDE Var  KDE Uni  Logn Uni  Logn Var  Norm Uni  Norm Var
         3-9            22.5     33.8      20.9       9.9       7.6       5.4
         10-99          20.0     22.5      25.2      17.9       8.8       5.6
         100-999        42.8     28.8       8.8      16.8       1.3       1.4
         1000+          69.6     23.5       0.6       6.1       0.0       0.2

     The kernel estimate under EQUAL weights leads at 3-9, the lognormal under
     equal weights leads at 10-99, and the kernel estimate under MARKET-SHARE
     weights leads everywhere above 100, reaching 69.6 percent.

     **THE CURVE IS U-SHAPED AND THE DIP IS NOT NOISE.** The kernel estimate is
     closer on 62.6 percent of datasets at n = 3-9, falls below half between
     about 10 and 55, and rises to 86.5 percent (equal weights) and 96.8
     percent (market-share weights) at the top of the corpus. With three to
     nine values there is no shape to estimate and both families do equally
     badly; between ten and fifty the lognormal's shape assumption is worth
     more than the kernel's flexibility. **The figure draws the dip rather than
     smoothing it**, because a monotone curve there would be the conclusion and
     not the data.

     **AND NO OTHER CHARACTERISTIC YIELDS A USABLE THRESHOLD.** Sweeping every
     candidate for the value at which the kernel estimate overtakes the
     lognormal, on the corpus: the modality index NEVER crosses under equal
     weights (60.9 to 55.9 percent as it rises) and crosses DOWNWARD under
     market-share weights (67.0 to 46.5), so more modality makes the kernel
     estimate relatively WORSE, which is the opposite of the intuition and
     survives holding dataset size fixed. Silverman's critical bandwidth is
     flat, 77.5 to 76.0 percent. **This confirms decision 88 on 10,000 datasets
     out of sample where that decision had 127 in sample.**

     **THE ONE LARGE NON-SIZE EFFECT IS NOT A DATA CHARACTERISTIC.** The
     variable-weighted mean of a dataset normalized to an unweighted mean of
     1.0 -- which is exactly how far market weighting shifts the average --
     crosses at 0.872 under equal weights (31.5 to 67.5 percent) and 1.17 under
     market-share weights (81.5 to 34.7). A practitioner cannot compute it
     without already knowing the shares, which is the same disqualification
     decision 88 applied to the uniform-to-variable distance. **So the rule a
     reader can act on still has exactly one number in it, and that number is
     how many declarations they hold.**

140. **2026-09-21, Stage 2f review. MARKET-SHARE WEIGHTING PAYS WHEN THE SHARES
     ARE CONCENTRATED, NOT WHEN THEY ARE EVEN, AND THAT IS THE OPPOSITE OF THE
     SAMPLE-SIZE INTUITION.** `[AUTHOR]` Asked whether the kernel estimate's
     loss under market-share weights at n = 3-9 was just noise. It is not, and
     answering it turned up the better finding.

     **AT n = 3-9 IT IS REAL AND THE MECHANISM IS THE POINT COUNT.** A flat
     Dirichlet over three to nine points leaves a Kish effective sample size of
     **2.7 in the median**, and **93.1 percent of those datasets have fewer
     than five effective observations**. The market-share fit beats its own
     uniform twin on 39.0 percent of them, interval [37.2, 40.9] over 2,500
     datasets, so it is far outside sampling error. With three to nine values
     there is no shape to estimate under any weighting and putting weights on
     them adds variance without adding information. Splitting that band by how
     much effective sample survives shows no gradient at all -- 41.4, 36.5,
     36.3, 41.9 percent across quartiles -- which is the confirmation: nothing
     about the weights rescues it, because the problem is the point count.

     **AT n = 100-999 THE SAME SPLIT INVERTS, AND STEEPLY.** Share of datasets
     where the market-share fit beats its uniform twin, by how concentrated the
     weight vector is:

         n_eff / n     kernel estimate   lognormal
         0.23 (most concentrated)  74.4        79.4
         0.33                      70.7        76.0
         0.42                      61.6        66.1
         0.51 (most even)          29.8        37.9

     **A nearly even weight vector carries no information about the market, so
     the market-share fit is the uniform fit plus noise and it loses.** A
     concentrated one says something, and it wins. Concentration is therefore
     not merely a shrunken sample; the usual intuition captures only the
     variance half and misses the signal half. `TABLE_ReductionWeightConcentration.csv`.

     **This is the clean statement of what variable weighting is FOR**, and it
     sharpens decision 97, which found at matched effective sample size that
     COHERENT concentration moves the answer 1.5 to 3.1 times as much as random
     concentration. Same mechanism from the other side: what pays is the
     weights being informative, not their being many.

     **AND IT IS MEASURED AGAINST THE MARKET-WEIGHTED PARENT, WHICH IS THE ONLY
     TARGET THAT CAN ANSWER IT.** Share of datasets above n = 1,000 where the
     market-share kernel fit beats its uniform twin: **75.8 percent** against
     `w1_market`, **20.7 percent** against `w1_parent`, **98.2 percent** against
     the in-sample `w1`. Under `w1_parent` the two schemes are graded against
     different populations and under `w1` every model is scored against the
     variable-weighted data it was fitted to. Decision 65 records why; this is
     the first place the three have been printed side by side, and the spread
     is the argument for doing so.

141. **2026-09-22, Stage 2f review. THE TWO ARMS DRAW MARKET-SHARE WEIGHTS BY
     DIFFERENT RULES, ON THE DIMENSION THE PAPER IS ABOUT. This is a real
     structural inconsistency, it is NOT fixed here, and Stage 2h owns it.**
     `[AUTHOR]` "Seems unfair to use a different weighting mechanism for
     empirical and synthetic."

     **WHAT THE TWO ARMS DO.** `empirical.py:389` draws a flat Dirichlet over
     all n points, with no structure, because real data has no known modes.
     `generator.py:329` gives each point `market[mode(i)] * within`, a
     mode-level share split inside the mode, at `mode_coupling = 1.0`. So the
     weights are correlated with the values on one arm and independent of them
     on the other.

     **WHY THAT MATTERS RATHER THAN BEING A DETAIL.** Weights drawn
     independently of the values are exchangeable, so the weighted CDF
     converges to the unweighted one and the measured weighting effect MUST
     decay like n^-1/2. Correlated weights do not decay. Median
     uniform-to-variable separation:

         n           empirical (flat)   synthetic (coupled)   corpus values, reweighted flat
         3-9              0.0831              0.1377                 0.1300
         10-99            0.1090              0.1248                 0.0860
         100-999          0.0737              0.0714                 0.0383
         1000+            0.0049              0.0501                 0.0116

     Decay slope on log(n): **-0.397 empirical against -0.167 synthetic.** The
     third column is the corpus's OWN values reweighted flat, which is what
     proves the gap is the weight rule and not the data. **At n >= 1000 the
     corpus shows ten times the empirical arm's weighting effect.** The
     apparent agreement in the aggregate is an accident of the size mix: 78 of
     147 real categories are n = 10-99, the one band where the two rules agree.

     **THE AUTHOR'S PROPOSED FIX, AND IT IS THE RIGHT DIRECTION.** Give both
     arms the same rule. A mixture model is NOT the way to do it -- it cannot
     be fitted at n = 3 to 9, mode counts on real data are badly
     method-dependent (95 percent unimodal at one bandwidth against 68 at
     another), and it would make the weights a function of a fitted model in
     the middle of the paper's central quantity. `weighting.block_weights`
     already does the job without any of that: cut the SORTED values into k
     contiguous blocks, draw block shares from a flat Dirichlet, split flat
     inside. What mode coupling actually produces is weights correlated with
     values; the mode structure is only the device.

     **A COHERENCE PARAMETER MAKES THE UNTESTABLE ASSUMPTION AN AXIS.** Rank
     the values to r in [0, 1], draw u uniform, sort by
     `s = rho * r + (1 - rho) * u`, cut into k contiguous blocks. rho = 1 is
     maximal clustering, rho = 0 is random membership. Measured median
     separation on the 147 real categories at n >= 1000: **0.0075 at rho = 0
     rising to 0.1219 at rho = 1**, against 0.0049 today -- a factor of 25 at
     the top end. Whether the effect survives large n, as the ratio of the
     1000+ median to the 10-99 median: empirical 0.05 / 0.16 / 0.28 / 0.41 /
     0.53 and synthetic 0.15 / 0.35 / 0.71 / 1.00 / 0.85 across
     rho = 0, 0.25, 0.5, 0.75, 1.

     **A BIGGER SEPARATION IS NOT EVIDENCE OF A BETTER MODEL, and the author
     asked exactly this.** The separation measures what unknown shares do; it
     is not a target. The factor of 25 comes from ASSUMING maximal clustering.
     What is a defect, and worth fixing whichever rho is defensible, is that
     the two arms differ at all.

     **BUT rho = 0 IS NOT THE NEUTRAL CHOICE, AND CALLING IT AGNOSTIC WAS
     WRONG.** The author's objection, and it is correct: "being agnostic is
     making a decision in the wrong direction ... saying that weight doesn't
     matter at high n is just a mathematical artifact, not a reflection of
     reality." A flat Dirichlet is not the absence of an assumption. It is the
     specific claim that market share is UNCORRELATED with carbon intensity,
     and that claim is almost certainly false: the 63.75 percent
     Rest-of-World BOF share of Marsh, Hattam and Allen (2025) sits on the
     HIGHER-carbon steel route, while the lower-carbon EAF route is the small
     one. Share tracking technology is exactly the correlation rho measures.

     **And the decay itself is a property of the model rather than of markets.**
     Weights drawn independently of the values must converge to uniform as n
     grows, because that is what exchangeability means. Real market share does
     not become more uniform as more manufacturers publish declarations. So the
     n^-1/2 decay this corpus shows at rho = 0 is an artifact of the weight
     model, and reporting it as a finding about weighting would be reporting an
     artifact.

     **WHAT THIS CHANGES FOR 2h.** The task is NOT "sweep rho and report the
     range". It is: sweep rho, REJECT rho = 0 explicitly as a null that the
     evidence contradicts, and anchor a primary value on the published
     production-volume data rather than on agnosticism. The symmetric framing
     -- both ends are equally assumptions -- is true and is not a reason to
     default to the end we can already see is wrong.

     **TWO CONFOUNDS A LATER STAGE MUST CONTROL.** The block model changes
     CONCENTRATION as well as coherence -- with k blocks the weight is
     concentrated into k groups whatever rho is, which is why rho = 0 already
     sits above today's flat draw -- so k must be matched or swept alongside,
     exactly as decision 97 matched the Kish effective sample size. And the
     empirical arm's decay is PARTLY DISPERSION: the categories above n = 1,000
     are the ReadyMix strength classes at a median coefficient of variation of
     0.297 against 0.954 at n = 100-999, and decision 96's law says separation
     scales with that, so some of the decay is concrete being tight rather than
     weighting dying.

     **WHAT IS DECIDED HERE: nothing about the weight model.** Mode coupling
     stands, the corpus is not reweighted, no number moves, and Stage 2h takes
     this as its first item with the numbers above. Decision 97 already
     assigned the block-structure sweep there. **Also handed to 2h:** the
     author's proposal to draw mode SIZES from a flat Dirichlet rather than at
     concentration 10, which is more realistic -- at k = 2 the largest mode
     spans 0.52 to 0.97 instead of 0.51 to 0.71 -- and which measured 5.0 seed
     standard deviations worse on the calibration, though that comparison used
     the mismatched weight rules above and must be redone once the arms agree.
     Discrepancy entry 130.


142. **2026-09-22, Stage 2f review. THE PRACTITIONER THRESHOLD IS 81
     DECLARATIONS AND EVERYTHING FROM 68 TO 97 IS INDISTINGUISHABLE FROM IT.
     This SUPERSEDES the "59 to 134" of decision 139**, which was too wide.
     `[AUTHOR]` "The question I'm more interested in is how precisely must I
     set a threshold. 134 still feels too high for that, right?" It was.

     **WHAT WAS WRONG WITH 59 TO 134.** `flat_region` calls a threshold as good
     as the best when its cost sits within five percent of the span between the
     best and the better fixed policy. Both the five percent and the span are
     choices of mine rather than facts about the data, and the answer carried
     no uncertainty at all, so it could not separate "really as good" from
     "looks as good". It was too permissive at the top.

     **THE INSTRUMENT.** `metricreduction.threshold_interval`, two bootstraps
     over datasets. The first resamples, refits the whole cost curve and takes
     its argmin, so its spread says how well the data pin the threshold down:
     **68 to 116**. The second is PAIRED -- each threshold's excess cost over
     whichever threshold won ON THAT SAME RESAMPLE -- so the variation common
     to both cancels and the interval is about the difference rather than the
     level. The thresholds whose interval reaches zero cannot be told apart
     from the best: **68 to 97**, with the optimum at **81**.

     **EIGHT INDEPENDENT BOOTSTRAP STREAMS RETURN 81, 68 AND 97 EXACTLY**, so
     unlike several numbers this stage has had to withdraw, these are not one
     draw from a distribution.

     **THE PENALTY FOR MISSING IT**, in points of extra error over the best
     choice that could be made per dataset, with 95 percent intervals:

         threshold    penalty    interval
            24          4.84     3.54 to 6.25
            48          1.83     1.01 to 2.83
            81          0.11     0.00 to 0.50
            97          0.27     0.00 to 0.73
           138          0.79     0.16 to 1.58   <- excludes zero
           304          6.47     5.00 to 7.93

     **TWO RANGES IN THIS STAGE ARE DIFFERENT QUANTITIES AND MUST NOT BE
     QUOTED FOR EACH OTHER.** The kernel and lognormal FAMILIES change places
     at **46 to 70** declarations, which is a property of the win-share curves
     and moves by a few on a reseed (46 to 49 low, 70 to 73 high, across five
     streams). The best place for a RULE is **68 to 97**, above the crossing
     because the penalty curve is steeper on the high side -- 4.8 points at a
     threshold of 24 against 6.5 at 304. `family_lead_curve` and
     `crossover_band` produce the first, `threshold_interval` the second.

     **A FIGURE BUG FOUND WHILE CHECKING THIS, and it is the kind that widens a
     claim quietly.** Figure A took the MIN AND MAX of the indistinguishable
     flag rather than its longest unbroken run, and one isolated threshold
     beyond a distinguishably worse one stretched the reported band from 68-97
     to 68-116. At the edge of a near-zero effect the flag jitters: 106 is
     excluded with a lower bound of 0.120 while 116 is included at 0.000.
     `metricreduction.longest_true_run` requires the run to be unbroken, and
     the real case is its test.

     **AND WHICH COMPARISON PRODUCES A CROSSING MATTERS MORE THAN THE CROSSING.**
     The market-share kernel estimate passes the better single lognormal at 65
     declarations, the kernel family passes the lognormal family at 46 to 70,
     and against the two lognormals pooled it does not pull clear until past
     200. An earlier draft of this stage quoted 65 without saying which
     comparison it came from, which is how a figure ends up disagreeing with
     the sentence beside it. Discrepancy entry 128.

143. **2026-09-22, Stage 2g. THE HEADLINE METRIC IS JUDGED BY WHETHER IT
     RECOVERS THE TRUTH, AND "ECI RANK #1 FREQUENCY" IS THE WORST OF SEVEN
     CANDIDATES.** `[AUTHOR]` Three earlier arguments said to demote it -- it is
     fragile with four exchangeable materials (Stage 2d), it carries a 3.67
     percent argmax noise floor (Stage 2e), and it is only 9 percent predictable
     from a material's own dataset because it belongs to the GROUP (Stage 2f).
     None of them is the deciding one. Stage 2e's run against the true parents
     means the question is no longer "is this sensitive" but "does this recover
     the right answer".

     **THE STATISTIC.** Mean absolute error against the true parent, divided by
     the standard deviation of the TRUE value across every material in the arm.
     Below 1 a method's error is smaller than the differences between materials
     the metric exists to reveal; at or above 1 the metric cannot distinguish
     two materials at all. It is the same division as the study's own NRMSE on a
     different numerator, so the two are directly comparable.

     **THE DIVISOR IS THE TRUTH'S SPREAD AND NOT THE METHOD'S OWN**, because
     dividing by the method's spread would let a method that reports nearly the
     same number for every material improve its score by being less
     informative. `tests/test_metricset.py` plants exactly that case and asserts
     the flat method loses.

     **THE RESULT, on 10,000 materials against the market-weighted parent, best
     method to worst:**

         spread of a material's contribution      0.42 to 0.50
         95th percentile of its contribution      0.45 to 0.51
         its estimated contribution               0.51 to 0.71
         the uncertainty index                    0.51 to 0.53
         its share at the BUILDING's 95th pct     0.61 to 0.69
         its mean share of the total              0.66 to 0.88
         ITS CHANCE OF LEADING                    0.72 to 1.07

     **The study's own headline is last on both ends, and under a normal fit it
     exceeds 1.0** -- 1.037 uniform and 1.074 variable -- so on that metric a
     normal fit's error is larger than the whole spread of true rank-1
     frequencies across materials and the metric carries no information about
     which material is which.

     **THE INSTABILITY IS CORROBORATED ON A REAL DESIGN AND IS NOT AN ARTIFACT
     OF THIS CONSTRUCTION.** Every material here is normalized to a mean of 1.0
     and carries a use intensity of 1.0, which makes a ranking as fragile as it
     can be made, and that is a fair objection to the numbers above. But Marsh,
     Lewis, Hattam and Allen (in press) report the same thing for a real
     four-option staircase: the top-contributing product changes with the
     uncertainty characterization scenario. **So it is a property of ranking
     near-equal contributors rather than of synthetic data, and the paper should
     report it plainly rather than defensively.**

     **AND WHICH METHOD LOOKS BEST DEPENDS ON WHICH METRIC IS REPORTED.** On a
     win share against the truth, `KDE, Variable` leads on the chance of
     leading, the 95th percentile and the spread; `Lognormal, Variable` leads on
     the mean contribution, the mean share and the share at the building's 95th;
     `Lognormal, Uniform` leads on the uncertainty index. **Three different
     methods across seven metrics**, and the rank correlation of the six
     methods' ordering with the ordering the rank metric gives runs from +1.00
     to **-0.54**. A paper that reports one metric and calls a method best is
     reporting the metric. `TABLE_MetricRecovery.csv`, `TABLE_MetricVerdict.csv`,
     `TABLE_MetricWinShare.csv`.

144. **2026-09-22, Stage 2g. WHAT THE PAPER SHOULD LEAD WITH, in the order the
     results section takes.** `[AUTHOR]` Stage 2e recommended the five
     statements and this confirms the order by measurement rather than by
     argument.

     **EVERY NUMBER IN THIS ENTRY IS PRE-REGENERATION AND PRE-WEIGHT-PORT AND
     MUST BE REBUILT FROM `TABLE_FiveStatements.csv` RATHER THAN PATCHED.** The
     ORDER this entry sets is what stands. Decision 201 is why the whole entry
     is flagged instead of the three numbers that are known to have moved: the
     ones nobody has checked are exactly the ones that get missed. The current
     values are in decision 252.

     1. **The design comparison**, because it is the decision a designer makes
        and the answer is a null: over 800 option pairs scored against the true
        distributions, the choice of UQ method changes the stated probability
        that a substitution is an improvement by at most **0.020**, and every
        method lands within **0.026** of the truth. At a claimed 5 percent
        saving the truth is 0.629 and the six methods span 0.630 to 0.642.
     2. **The safe-lead rule**: the chance that the choice of method changes
        which material leads crosses 1 percent at a top-two contribution ratio
        of **2.13** [2.09, 2.17], and the one real building element available
        sits at **1.02**.
     3. **The building total and the budget statement**: every method
        understates the total's 90th percentile, by **0.087 to 0.356** on a
        building averaging about 4.0, and at a budget the truth meets 90.0
        percent of the time the six report **86.8 to 91.3** percent.
     4. **The specification result**, which is where the choice costs most:
        against a true mean saving of **5.39 percent** of the building the six
        report 4.88 to 6.19 percent, and asked for the chance of achieving at
        least 5 percent the truth is **23.2** percent while the six span
        **22.9 to 30.5**. Beside it, the QUANTITY reduction is 0.0625 of the
        building under every method and under the truth, identical to four
        decimal places, because it is a deterministic fraction of a material's
        own contribution and no distributional assumption enters.
     5. **Where the uncertainty sits**, the uncertainty index, which appears in
        no table, figure or section of this study so far. See decision 146.

     **The attribution metrics are ONE of the five and not the frame.** Report a
     material's estimated contribution as the primary magnitude, its share at
     the building's 95th percentile as the companion, and the chance of leading
     as a secondary statistic quoted with its noise floor.
     `TABLE_FiveStatements.csv`.

145. **2026-09-22, Stage 2g. THE MAGNITUDE COMPANIONS, AND THE NEW ONE IS NOT
     THE OLD ONE UNDER ANOTHER NAME.** `[AUTHOR]` Two were asked for: each
     dataset's mean share of the total, which already existed as
     `eci_perc_mean` and which no stage had compared against the rank metric,
     and its share at the 95th percentile of the total, which is new.

     **`eci_perc_p95tot` is each material's share of the building total in the
     iterations where the BUILDING sits at its 95th percentile**, read over a
     window of plus or minus 0.01 in quantile units, which is 200 of the study's
     10,000 iterations. It is the attribution question asked at the end of the
     distribution a carbon budget is written against.

     **IT IS NOT `eci_p95`**, which is the 95th percentile of a material's OWN
     contribution over its own marginal, and the iteration that puts one
     material at its 95th percentile is usually not the iteration that puts the
     building at its 95th.

     **AND IT IS NOT THE MEAN SHARE EITHER, which had to be checked rather than
     assumed.** Across the 60,000 rows the two correlate at only **0.317**, with
     a mean absolute difference of **0.061** where the average share is 0.25 and
     a maximum of **0.569**. A planted test where one material is the only
     source of the building's upper tail gives it a mean share below 0.35 and a
     share at the total's 95th percentile above 0.85.

     Adding it consumed no randomness: 36 of the 40 columns of
     `TABLE_PLCAResults.csv` are bit-identical to the previous run and the three
     new columns are pure additions. Decision 147 names the four that moved.

146. **2026-09-22, Stage 2g. THE UNCERTAINTY INDEX IS THE STEADIEST OUTPUT AND
     NO METHOD RECOVERS IT WELL, and reporting only the first half would be the
     most misleading thing this study could do.** `[AUTHOR]` Stage 2e
     recommended promoting it on the strength of its NRMSE between methods,
     0.503 against 1.042 for a material's chance of leading. That holds -- it is
     0.5035 [0.4918, 0.5158] here -- and it is only half of what a metric has to
     answer for.

     **FIRST, A CORRECTION TO AN EARLIER CLAIM THAT THIS STAGE REPEATED BEFORE
     CHECKING. It originates in Stage 2d's decision 103 -- "the study computes it
     as `ui` and reports it nowhere" -- and is repeated in Stage 2e's decision
     114 and in discrepancy entry 95. THAT IS FALSE and the author caught it.** The manuscript reports it in Figure 5b
     and 5d, defines it in Supplement 3(c), and draws a conclusion from it in as
     many words: "The NRMSE for the uncertainty index is much smaller than that
     for the ECI Rank #1 Frequency, indicating that different UQ methods result
     in similar uncertainty indices", and again in the conclusions, "pLCA
     results related to the variance of total embodied carbon, such as the
     uncertainty index, did not differ substantially between UQ methods."

     **What is true is narrower and is still worth acting on.** It is reported
     as a secondary observation about NRMSE rather than as one of the questions
     a probabilistic LCA answers. The recommendation is to PROMOTE it to one of
     the five headline categories -- "which material drives the uncertainty in
     the total" -- not to introduce it. Discrepancy entry 145.

     **The other half: every one of the six methods is out by about half the
     spread between materials.** Recovery error 0.508 to 0.531, a span of 4.4
     percent from best to worst. So the choice of method genuinely does not
     matter for it, which is the case for reporting it, and no method gets it
     right, which is the caveat that has to travel in the same sentence.

     **IT IS ALSO THE BEST OF THE SEVEN ON THE DECISION READING.** Asked which
     material's uncertainty dominates, the best method names the truth's answer
     **58.4 percent** of the time against a one-in-four chance level, which is
     the highest agreement of any candidate -- effectively tied with the spread
     of a material's contribution, also 58.4 percent -- against 52.5 percent for
     the chance of leading and 49.1 for a material's estimated contribution.

     **AND IT IS NOT IMMUNE TO THE TAIL FAILURE MODE.** Its tail exposure is
     1.58 and a thousandth of a model's mass at a thousand times the dataset
     mean moves it by 145 percent. It is a variance share, so one enormous
     material takes all the variance and the index follows.

     **THE PAIRING IS THE GENERAL LESSON AND IT IS WHY THE TWO STATISTICS ARE
     REPORTED TOGETHER.** A low NRMSE beside a high recovery error is a metric
     every method agrees on and every method is wrong about. `metric_verdict`
     joins them for exactly that reason.

147. **2026-09-22, Stage 2g. THE `(1 - capecc)` DIVISOR IS SETTLED: divide by
     the iterations in which the strategy APPLIES, and report the applicability
     instead of assuming it. This COMPLETES decision 116 and supersedes its
     final form.** `[AUTHOR]` The column carried an inline comment conceding
     that "percentages look off because not all reduction strategies apply in
     all scenarios". The comment named the right problem and the divisor was the
     wrong fix.

     **THE HISTORY, because three stages touched it.** It was a constant
     1 / 0.25, exact only while the specification cap was each METHOD'S OWN 75th
     percentile and therefore bound in exactly a quarter of iterations for every
     material by construction. Stage 2e made the cap an absolute value per
     material (decision 115) and had to drop the divisor with it (decision 116),
     leaving a plain count over all iterations whose four columns summed to
     between **0.31 and 1.00** rather than to 1.

     **THE CORRECT DENOMINATOR.** The question is "if I can pursue one
     specification cap, which material should I cap", and in an iteration where
     no cap binds there is no answer: every choice delivers nothing. Counting
     those iterations against all four materials makes the columns depend on how
     often the strategy applies rather than on which material is the right one.
     `capecc_rank_1` now sums to **exactly 1.0** across the materials of a pLCA,
     like every other rank-1 frequency in the study, and across the four ranks
     of one material it sums to the share of applicable iterations in which that
     material's own cap bound.

     **THE APPLICABILITY IS REPORTED AND NOT DIVIDED AWAY, and it is signal.**
     `capecc_applies` is the share of iterations in which any cap binds and
     `capecc_binds` the share in which this material's own does. The first runs
     from **0.727 under `Lognormal, Uniform` to 0.838 under `Normal, Uniform`**
     and has an NRMSE between methods of **1.204**, the highest of any cap
     column: a method that puts more mass above the cap finds it binding more
     often, and under the old constant divisor that was forced to 0.25 for every
     material and every method.

     **WHAT MOVED.** Only the four `capecc_rank_*` columns of
     `TABLE_PLCAResults.csv`; 36 of the 40 shared columns are bit-identical.
     `capecc_rank_1` mean **0.1929 to 0.2500**, `capecc_rank_2` 0.0922 to
     0.1167, `capecc_rank_3` 0.0238 to 0.0294, `capecc_rank_4` 0.00255 to
     0.00306. **The manuscript's definition needs one sentence changed**: it
     says these are "the percentage of iterations in which a given dataset is
     1st, 2nd, 3rd, and 4th", which is true of the quantity-reduction columns
     and not of the cap ones, whose denominator is the applicable iterations.

     **The same stale comment sat on the material-reduction block and was never
     true there**: that strategy applies in every iteration, so its columns are
     a complete ranking summing to 1 both ways. Removed. Discrepancy entry 7 is
     resolved.

148. **2026-09-22, Stage 2g. THE CONCLUSIONS DO NOT ALL SURVIVE THE COMPANIONS,
     AND "THE NORMAL IS FORTY PERCENT WORSE" IS A STATEMENT ABOUT ATTRIBUTION
     RATHER THAN ABOUT EVERYTHING A pLCA SAYS. This NARROWS decision 109.**
     `[AUTHOR]` Adding a companion metric is only worth something if the paper's
     claims are re-read against it, and three of them do not survive.

     How much worse the better of the two normal fits is than the best of the
     four non-normal methods, on recovery against the true parent:

         a material's chance of leading         44.2 pct   normal is worst
         its estimated contribution             36.5 pct   normal is worst
         its mean share of the total            27.6 pct   normal is worst
         the spread of its contribution         17.9 pct   normal is worst
         the 95th percentile of it               4.4 pct   normal is NOT worst
         its share at the building's 95th        2.8 pct   normal is NOT worst
         the uncertainty index                   1.0 pct   normal is NOT worst

     **On three of the seven the normal is not the worst method and is within
     four percent of the best.** The ordering of the six methods is not stable
     either: its rank correlation with the ordering the chance of leading gives
     runs +1.00 on the mean share, +0.77 on the spread, +0.49 on the mean
     contribution, and **-0.26, -0.31 and -0.54** on the 95th percentile, the
     share at the building's 95th and the uncertainty index.

     **WHAT DOES SURVIVE EVERY METRIC IS THE BIAS, and it is the part that
     matters at building scale.** On the three metrics where a signed error is
     informative -- the mean contribution, the 95th percentile and the spread --
     the normal is the MOST biased of the six on all three, by **+0.21, -0.26
     and -0.44** in units of the metric's own between-material spread against
     the kernel estimate's +0.01, -0.11 and -0.22. Bias adds across the
     materials of a building while noise cancels, which is decision 122b. **On a
     share or a rank frequency the signed error is identically zero by
     construction**, because the four values sum to one, so that column carries
     no information for those metrics and must not be read as evidence of
     unbiasedness.

     **So the paper's sentence has to name the statement.** "Do not fit a normal
     distribution" is right for attribution and for anything a building total is
     summed from, and it is not supported by the tail and information metrics,
     where the normal is as good as anything and merely more biased.
     `TABLE_MetricConclusions.csv`.

149. **2026-09-22, Stage 2g. THE TAIL A GOODNESS-OF-FIT SCORE CANNOT SEE: the
     body of W1 charges for the MASS a model misplaces and not for the DISTANCE,
     and the tail term Stage 2c added is what closes it.** `[AUTHOR]` Stage 2b
     handed this stage the downstream end of the question: a statistic between
     CDFs is nearly blind to tail mass and a Monte Carlo is not, because it
     samples. It was checked rather than assumed, and the check is sharper than
     the question.

     One material of a real pLCA group has a fraction of its fitted model's mass
     moved to 10, 100 or 1,000 times the dataset mean, with the other three
     models, the uniform variates and the group all held.

     **W1 TAKEN OVER THE SCORING GRID ALONE GOES BLIND THE MOMENT THE MASS
     LEAVES THE GRID, AND THE GRID ENDS JUST PAST THE DATA.** Its top is
     `max(x) + 10 sd`, which on the group measured is **9.902 times the dataset
     mean**. The distance is swept continuously over 25 log-spaced points from 1
     to 3,000 times that mean, and at a thousandth of the mass **all 16 points
     beyond the grid give the same body score, 0.262571, to the last digit**,
     against an uncontaminated 0.254474. Inside the grid it does move, so it is
     charging for the contamination it can see and for nothing further. **With
     the tail term the same 16 points run from 0.2729 to 3.268, a factor of
     twelve.**

     An earlier version of this entry reported three round decades, which could
     not show WHERE the blindness starts and made it look like a property of
     large distances rather than of the grid's own edge.

     **AND THE METRICS SPLIT BY WHETHER THEY HAVE A CEILING.** Relative change
     over the 17 distances beyond the grid, first to last:

         spread of a material's contribution   0.203   -> 104.8
         the uncertainty index                 0.215   ->   1.46
         its estimated contribution            0.013   ->   1.99
         its share at the building's 95th      0.00823 ->   0.00823
         its mean share of the total           0.00203 ->   0.00246
         its chance of leading                 0.00129 ->   0.00129

     **A share and a rank frequency saturate: once a material's draw is enormous
     it holds the whole share and takes rank one, and making it two hundred
     times more enormous changes neither to the last digit.** A mean, a standard
     deviation and a variance share have no such ceiling; the spread of a
     material's contribution moves by a factor of 517 over that range.

     **THE CONSEQUENCE FOR THE RECOMMENDATION, and it is a tension rather than a
     clean answer.** The metrics that recover the truth BEST are levels, and
     levels are exactly what a thin far tail wrecks; the metrics that are immune
     are shares, and they recover worse. **What makes the levels safe to report
     is that the study's criterion now charges for the thing that wrecks them**,
     which it did not before Stage 2c. So `W1_TAIL_TERM` is not an accuracy
     refinement, it is the guard under every level metric this paper reports,
     and **Stage 2h must keep it in force and report `model_sd_ratio` at every
     value of `PROFILE_DELTA_LO_FRAC` it sweeps**, as decision 69 already
     requires.

     One caveat on the 95th percentile of a material's own contribution: its
     immunity is conditional on the contamination being thinner than 5 percent
     of the mass, because above that the misplaced mass is inside the quantile
     being read. `TABLE_MetricTailStress.csv`, `TABLE_MetricTailExposure.csv`.

150. **2026-09-22, Stage 2g. NOTEBOOK 3 STOPPED READING THE TARGET STAGE 2c
     RETIRED, and it was the last place in the study that still did.**
     `[AUTHOR]` One cell scored every fitted model by W1 against the
     variable-weighted empirical CDF of the values it had been fitted to, and
     the cell below it explained every pLCA outcome with that number. The target
     is circular twice over: it is the training data, and it is the
     variable-weighted curve, so a variable-weighted method is scored against
     itself.

     It now reads `w1_market`, the score against the market-weighted true
     parent, from the table notebook 2 writes. That is the population a
     probabilistic LCA of what gets built is a statement about and the only
     target under which all six methods estimate the same thing. The retired
     score is kept beside it, unused, and the cell prints how far the two
     disagree, so the change is measured rather than asserted.

151. **2026-09-22, Stage 2g. UNIFORM, TRIANGULAR AND BETA DO NOT BELONG IN THIS
     PAPER'S COMPARISON, AND UNIFORM AND TRIANGULAR DO BELONG IN STAGE 2h'S
     JUDGMENT ARM. This EXTENDS decision 124 rather than opening a new
     question.** `[AUTHOR ASKED]` The author asked whether other distributions
     are worth comparing and what else is common in LCA.

     **WHAT IS ACTUALLY USED.** The lognormal is dominant: it is ecoinvent's
     default and it is what the pedigree matrix produces, since a geometric
     standard deviation IS a lognormal parameterization. The normal is common
     and usually wrong for a strictly positive right-skewed quantity. Both are
     in the study. Uniform and triangular are used, and gamma, Weibull and beta
     appear occasionally -- beta for bounded quantities such as efficiencies and
     shares, which an embodied carbon coefficient is not.

     **WHY UNIFORM AND TRIANGULAR ARE NOT COMPETITORS HERE, and it is a
     question of what they are FOR rather than of how they perform.** This paper
     compares ways of turning a SET of declarations into a distribution. A
     uniform is not fitted to a dataset; it is specified from two numbers, and
     its maximum likelihood fit to n values is exactly [min(x), max(x)], which
     discards every value in between. A triangular adds a mode and discards the
     rest. They would lose the comparison by a distance, and **that is the
     reason not to include them**: a family that cannot use the data is a straw
     man, and a straw man that flatters this paper's own method is worse than
     no comparison at all.

     **WHERE THEY DO BELONG IS EXACTLY WHERE THE PEDIGREE MATRIX BELONGS.**
     Uniform and triangular are what a practitioner reaches for when there is no
     dataset -- a plausible low and high, perhaps a most likely value -- which
     is the same situation the pedigree matrix is built for. Decision 124 already
     records the framing that makes such a model comparable: this study's
     yardstick, how far apart two models have to be before the answer changes,
     does not care how either model was built. **So Stage 2h's judgment-driven
     arm should hold three things and not one: the pedigree matrix swept over its
     geometric standard deviation, a uniform over a plausible range, and a
     triangular over a range with a mode.** The question each answers is the
     same: how far from the data-driven answer does a judgment-driven model sit,
     and is that far enough to change the decision.

     **GAMMA IS ALREADY DONE AND WEIBULL IS ALREADY SCHEDULED.** Out of sample on
     the real categories the three-parameter lognormal is indistinguishable from
     gamma -- every paired bootstrap interval straddles zero -- and on the
     synthetic arm against the known parent it separates by +0.0117 and +0.0045,
     winning 77.4 and 67.5 percent of datasets, so it is never worse and it
     stands (decision 70). Weibull is on Stage 2h's list.

152. **2026-09-22, Stage 2g. THE RUNAWAY TAIL IS RARE AND REAL IN THIS STUDY'S
     OWN FITS, TWO GUARDS ALREADY CATCH IT, AND AN UPPER TRUNCATION IS THE EASY
     FIX THIS PAPER STATES RATHER THAN IMPLEMENTS.** `[AUTHOR ASKED]` The author
     asked whether bad tails make much difference here and whether truncation is
     worth implementing or worth naming as a weakness with an easy fix. The
     second, and here is the measurement that decides it.

     **IT IS NOT HYPOTHETICAL.** `model_sd_ratio`, the fitted model's own
     standard deviation over the data's, is near 1.0 in the median for every one
     of the six methods on both arms -- so the typical fit is fine -- and
     **0.25 percent of fits exceed five times the data's spread**. The worst
     single fits are `Lognormal, Uniform` at **73.7** times on the synthetic arm
     and 7.4 on the real one, `KDE, Uniform` at 50.1 and `Normal, Uniform` at
     40.6. **Every one of the three is an EQUAL-WEIGHTED fit at small n**; the
     market-weighted twins top out at 5.0, 1.4 and 1.0.

     **AND THE TWO GUARDS ALREADY IN PLACE ARE WHY IT DOES NOT REACH THE
     RESULTS.** The profile-likelihood guard on the lognormal threshold bounds it
     at fitting time (decision 51, which records a model reaching a standard
     deviation of 3,281 when that guard was set too loose), and the tail term in
     the scoring criterion charges for whatever survives. Measured: the part of
     the score lying BEYOND the grid is **exactly zero for the kernel estimate
     and for the normal**, which put no mass there at all, and averages 6.0e-5
     and 9.4e-5 for the two lognormals with a maximum of 0.076.

     **WHY AN UPPER TRUNCATION IS NOT IMPLEMENTED HERE.** Every model in this
     study is already truncated BELOW at zero, because a negative emission
     coefficient is not admissible, and that bound is external and needs no
     argument. An upper bound has no equally external anchor: the mass ceiling of
     100 kgCO2e/kg this study applies to the DATA (decision 49) is a physical
     bound in the data's own units, and every dataset here is normalized to a
     mean of 1.0, so it is not a fixed multiple. Choosing a multiple is a
     methodological decision with numbers attached to it, which is a sweep rather
     than a metric stage's business.

     **SO THE PAPER SAYS THIS, IN ONE SENTENCE, AND STAGE 2h MAY SWEEP IT.** A
     goodness-of-fit distance between cumulative curves cannot see how far out a
     model puts its rare values, so a model can score well and still wreck a
     Monte Carlo; this study charges for it with a tail term and watches it with
     the fitted-model spread ratio, and truncating each fitted model at a
     plausible multiple of the largest observed value would remove the failure
     mode outright at the cost of one more assumption.

153. **2026-09-22, Stage 2g review. THE FIGURE GUIDE WAS INCOMPLETE AND THE
     FIGURES BUILT TO IT WERE UNREADABLE. A takeaway title needs a subtitle
     saying what is plotted, and an axis label is a noun phrase and not a
     sentence.** `[AUTHOR]` The author's words, on three figures built to
     `FIGURE_STYLE.md`: "I'm really starting to regret telling you to follow
     Jean Luc Doumont's advice for using strong takeaways as titles ... you're
     so bad at coming up with strong takeaways as titles. They're so vague, have
     no description, and leave you completely in the dark about what's actually
     shown in the plot. And axis labels keep getting drawn out to 2-3 sentences
     rather than just a succinct, clear label."

     **THE FAULT IS THE GUIDE'S AND IT IS NOW FIXED THERE.** Section 1 said the
     title carries the message and said nothing about where the DESCRIPTION
     goes. A writer following it puts the message in the title and then has
     nowhere to say what is on the axes except the axis label, which becomes a
     paragraph. Three slots, three jobs: **title** the message, **subtitle**
     small and gray under it saying what is plotted, **axis label** a short noun
     phrase with units.

     **AND A TAKEAWAY TITLE IS NOT A LICENCE TO HIDE THE DATA.** The author on
     the figure that was cut: "Why are we showing a range without labeling the
     UQ methods? We have a color coding system for which UQ method is which,
     what's the purpose of hiding that?" It showed each metric's recovery error
     as a best-to-worst range with no method named, vague row labels, and a
     title that asserted a recommendation the panel could not support.
     **`CompareUQMethods_FIG_MetricChoice` is deleted**, cell and PNG, and its
     one unique content is now the gray half of the scorecard's stacked bar.

154. **2026-09-22, Stage 2g review. THE SCORECARD IS THE STAGE'S FIGURE, and
     three changes made it readable.** `[AUTHOR]` "I have a feeling this figure
     will succinctly capture every single point we make in this stage if we do
     it right."

     **ONE SHARED COLOR SCALE, not one per row.** The first version normalized
     each row to itself, so 35.5 percent and 8.0 percent came out the same
     shade: "Why aren't these on the same scale? 35.5 has the same color as 8.0
     and 5.7." The best method in each row is boxed instead, which keeps the
     within-row reading without lying about the between-row one.

     **CONTINUOUS VALUES, NOT RANKS 1 TO 6.** "Can we show more continuous
     values for each of these so it's easy to tell how close it is?" A rank says
     nothing about whether second place is a hair behind the best or twice as
     wrong. Each cell is now the method's excess over the BEST method on that
     row, in percent of the size of the thing being claimed.

     **AND THE BAR IS STACKED, WHICH IS THE MOST INFORMATIVE THING IN THE
     STAGE.** The worst method's total error, split into what the best method
     still gets wrong and what the choice of method adds. Without the first half
     a claim where every method is badly wrong looks identical to one where
     every method is right. Measured: on a material's chance of being the
     largest contributor the total is **107 percent** of the spread between
     materials and **72 of it is there under the best method too**; on how often
     a specification cap binds the total is 31 percent and almost all of it is
     the choice. **So for most claims a better method moves the answer closer to
     the truth rather than to it.**

155. **2026-09-22, Stage 2g review. THE FIVE QUESTIONS ARE THE FRAME, AND
     ATTRIBUTION IS "WHICH MATERIAL IS BIGGEST, AND HOW OFTEN".** `[AUTHOR]`
     "Let's make sure we're framing this around the major categories of
     takeaways a user can glean from a probabilistic LCA." Decision 114 named
     five statements; this makes them the organizing structure of the results
     and of the scorecard, in a reader's words rather than in the study's:

         what is the building's total embodied carbon?      magnitude
         which materials contribute most, and how often?    attribution
         which materials contribute most to the UNCERTAINTY? information
         how effective is a reduction strategy?             action
         is this design better than that one?               comparison

     **Attribution is deliberately two things in one question**, because "which
     material is biggest" and "how often is it biggest" are the same question
     asked as a point estimate and as a probability, and the study reports six
     numbers that sit between them.

     **THE COST OF CHOOSING A METHOD, BY QUESTION**, on the claim that costs
     most in each, as a percentage of that claim's own size: attribution
     **35.5**, action **30.6**, magnitude **4.9**, information **2.2**,
     comparison **0.8**. The two questions a designer acts on most directly are
     the two the choice affects least.

     **AND THE ACTION HEADLINE IS EFFECTIVENESS, NOT APPLICABILITY.** An earlier
     draft quoted "how often a specification cap applies" as the action
     question's cost, and the author objected: "Shouldn't we be more concerned
     about the calculated effectiveness of that strategy with different UQ
     methods?" Correct. Applicability is the MECHANISM behind the spread and is
     reported as a diagnostic; the claim is what the strategy delivers.

156. **2026-09-23, Stage 2g review. THE SCORECARD SHOWS EACH METHOD'S TOTAL
     DISTANCE FROM THE TRUTH, NOT ITS EXCESS OVER THE BEST METHOD. This
     REPLACES the cell quantity chosen in decision 154.** `[AUTHOR]` "Rather
     than Figure B showing extra error over the best method, shouldn't it just
     show the total error relative to the parent distribution?" Yes, and the
     reason is a gap the author named exactly: "it's not clear to me that, when
     it comes to attribution, how close the 'right' method is to the parent
     distribution. That seems like an important detail here."

     **IT IS THE DETAIL, AND THE EXCESS FRAMING HID IT.** The two numbers answer
     different questions. The SPREAD between the best and worst of the six says
     what the choice of method costs; the BEST METHOD'S OWN ERROR says whether
     the answer is any good at all. A claim can have a small spread because
     every method is right or because every method is wrong, and only the second
     number separates them.

     **THREE SITUATIONS THE EXCESS FRAMING MADE LOOK ALIKE**, all in units of
     the size of the claim:

         claim                          choice costs   best method is off by
         a material's chance of being
           the largest contributor          35.5              71.9
         how often a cap binds              31.0               0.5
         the uncertainty index               2.2              50.8

     On the first, picking well is about a third of the problem and the rest is
     there whatever you do. On the second, picking well is nearly the whole
     problem. On the third, picking makes no difference and none of the six is
     close. **`total_error` is the new column and it is what the cells show;
     `excess` and `stakes` are kept, and `stakes` is the right-hand bar.**

     **THE PARAGRAPH THAT FOLLOWED THIS ONE IS SUPERSEDED BY DECISION 157.** It
     kept two denominators on one figure and printed each under its question,
     on the reasoning that naming them made them comparable. It does not: they
     are two different statistics, one a signal-to-noise ratio and one a
     relative error, and a shared color scale over both compares unlike things
     however they are labeled. Every row now divides by the true LEVEL. The
     original paragraph follows unchanged.

     **AND EACH QUESTION NOW NAMES ITS OWN DENOMINATOR ON THE FIGURE**, because
     the five do not share one and a shared color scale without that is
     misleading. A material's mean contribution is 1.0 for every material by
     construction -- every dataset is normalized to an unweighted mean of 1.0,
     decision 6 -- so the only meaningful scale for an attribution claim is how
     much the number varies BETWEEN materials; a building total has a level of
     its own and is scaled by it. **A cross-question comparison of these
     percentages therefore depends on that choice of denominator and the paper
     must say so**, which is why the denominator is printed under each question
     rather than left in a caption.

157. **2026-09-23, Stage 2g review. EVERY ROW OF THE SCORECARD NOW DIVIDES BY
     THE TRUE LEVEL, AND THE TWO-DENOMINATOR SCHEME OF DECISION 156 IS
     WITHDRAWN. The percentages on that figure were two different statistics
     wearing one unit.** `[AUTHOR]` "Why does it matter if the denominators are
     different in the figure? They're all expressed as percentages - isn't that
     effectively the same unit? I'm open to discussion here, because I want to
     be careful that we're not comparing numbers that shouldn't be compared ...
     Wouldn't expressing these in percent error mean they're comparable?"

     **THE ANSWER IS NO, THEY WERE NOT THE SAME UNIT, AND YES, PERCENT ERROR
     FIXES IT.** Seven rows -- the six attribution claims and the uncertainty
     index -- divided the mean absolute error by the SPREAD of the true value
     across materials; the other ten divided it by the true LEVEL. Error over
     spread is a signal-to-noise ratio and error over level is a relative error,
     and the percent sign made them look alike.

     **AND THE TWO ARE NOT A FIXED MULTIPLE OF EACH OTHER, so the seven rows
     were not comparable with each other either.** Level divided by spread, on
     the market-weighted truth run:

         a material's share of the total        6.57
         its mean contribution                  4.39
         its share at the building 95th         2.70
         its 95th percentile                    2.43
         its chance of being largest            2.25
         its standard deviation                 1.54
         the uncertainty index                  1.17

     A third inconsistency sat inside the magnitude block: all four of its rows
     divided by the true building TOTAL, so the error in the total's standard
     deviation was expressed as a fraction of the total's MEAN.

     **THE ONE DEFINITION, and it is available for every row**: the mean
     absolute error against the true parent, divided by the mean TRUE LEVEL of
     the same quantity. Every attribution number has a well-defined,
     non-degenerate level -- 1.0397 for a material's mean contribution, 0.6200
     for its standard deviation, 2.0812 for its 95th percentile, 0.2500 for each
     of the three shares and frequencies and for the uncertainty index -- so
     nothing has to be dropped for want of a denominator.

     **IT IS A RATIO OF MEANS AND NOT A MEAN OF RATIOS, which is the ordinary
     mean absolute percentage error and is NOT usable here.** The true
     uncertainty index reaches **-0.000671** and **2,904 of 60,000** materials
     carry a true value below a hundredth of the mean, so a per-material ratio
     is unbounded and, where the truth is negative, signless. The ratio of means
     is stable, is defined on every row, and is what the magnitude and action
     rows were already computing.

     **MEASURED 2026-09-23, after the author asked whether the natural per-pLCA
     reading is what is computed. It is, on most rows, and the exception is the
     reason for the choice.** Per method, `KDE, Uniform`:

         claim                      ratio of means   mean of ratios   median of ratios
         a material's contribution       12.63            12.88             6.23
         its chance of being largest     32.60            53.04            24.99
         the uncertainty index           45.04           134.81            40.57

     On the magnitude claims the two forms agree to a couple of tenths, so the
     author's reading of the figure is right there. On the uncertainty index
     the mean of per-case ratios is 135 percent and its MEDIAN is 40.6, next to
     the ratio of means at 45.0 -- which shows the 135 is a handful of
     near-zero denominators rather than typical performance.

     **AND IT MUST BE THE ABSOLUTE ERROR.** The mean SIGNED percentage
     difference is exactly 0.00 for every share and every rank frequency,
     because the four materials' values sum to one and the errors cancel by
     construction. A signed reading would report the uncertainty index as
     perfect.

     **`total_w1` IS DROPPED and the scorecard is 16 claims, not 17.** It is the
     Wasserstein distance between the method's building total and the truth's,
     so its true value is zero by definition and it has no level to be a
     percentage of. It stays in `TABLE_PLCABuildingSummary.csv`.

     **BOTH STATISTICS ARE KEPT, IN TWO PLACES, BECAUSE THEY ANSWER DIFFERENT
     QUESTIONS.** `recovery` in `TABLE_MetricRecovery.csv` divides by the spread
     and answers "can this metric tell two materials apart under this method",
     which is what ranks a CANDIDATE METRIC and is what section 1 of the handoff
     reports. `rel_error`, new in the same table, divides by the level and
     answers "how wrong is this number", which is what compares one CLAIM with
     another and is what the scorecard draws. `tests/test_metricset.py` pins the
     distinction and pins why the mean-of-ratios form is not used.

158. **2026-09-23, Stage 2g review. THE TAIL FIGURE IS CUT. The finding is a
     paragraph and two tables, and decision 149's numbers stand unchanged.**
     `[AUTHOR]` "Figure C I'm still on the fence about. I'm not sure it earns
     its place. What point is it supposed to make? 'W1 stops charging' doesn't
     make any sense. What does charging mean in this context? ... Are we making
     a whole figure devoted to what extreme values do to a dataset? That doesn't
     seem super important since you can just truncate those out."

     **THE JARGON WAS THE WRITER'S FAULT.** "Charging" meant "adding to the W1
     score" and the two lines were W1 with and without the tail term Stage 2c
     added. A title needing that much explanation is not a title.

     **AND THE SUBSTANCE DOES NOT CARRY A MAIN-TEXT FIGURE.** It is a stress
     test, not an observation: across 60,000 fits on both arms the mean charge
     for mass beyond the scoring grid is 0.0000 to 0.0001 and a fraction of a
     percent of fits exceed five times the data's own spread, because the
     profile-likelihood guard of decision 51 already bounds it. The author's own
     point settles the rest -- an upper truncation removes the failure mode
     outright, and Stage 2h already owns it.

     **WHAT SURVIVES, AND IT IS ENOUGH.** The measurement stays in
     `TABLE_MetricTailStress.csv` and `TABLE_MetricTailReality.csv` and is
     printed by two notebook cells: scored over the grid alone the criterion is
     flat past the grid's top at 9.9 times the dataset mean, the same 0.262571
     at all 16 distances beyond it, while a material's standard deviation moves
     by a factor of 517 over the same range. So the criterion charges for the
     MASS a model misplaces and not for how far out it puts it.
     `CompareUQMethods_FIG_TailBlindSpot` is deleted, cell and PNG. **Decision
     149 is unchanged**; only its figure is gone.

159. **2026-09-23, Stage 2g review. THE UNCERTAINTY INDEX IS BOTH THINGS AT
     ONCE, AND THE EXPLANATION IS DATASET SIZE: nine tenths of each method's
     error is an error ALL SIX make, and it is the error of estimating a
     variance from a handful of declarations.** `[AUTHOR]` "For the uncertainty
     index, the fact that the best is off by 43.6 percent doesn't make any
     sense to me. I thought that was the most consistent value across UQ
     methods -- why would they disagree so significantly but so similarly
     against the parent distribution?" The two facts are not in tension and the
     measurement says why.

     **FIRST, WHAT THE TRUE VALUE LOOKS LIKE.** `ui_j` is
     `Var(material j) / Var(total)`, a variance share, so its mean is 0.25 with
     four materials by construction. It is NOT a stable number near 0.25: across
     the 10,000 materials the TRUE value runs from **0.009 at the 10th
     percentile to 0.558 at the 90th**, with a standard deviation of 0.214 and a
     maximum of 0.984. So 43.6 percent of 0.25 is an absolute error of
     **0.109** on a quantity that genuinely spans almost nothing to almost
     everything.

     **SECOND, THE ERROR IS SHARED RATHER THAN METHOD-SPECIFIC.** Decomposing
     each method's error into the part common to all six and the residual:

         method                mean |error|   shared part   method-specific
         KDE, Uniform             0.1126        0.1017          0.0395
         KDE, Variable            0.1089        0.1017          0.0380
         Lognormal, Uniform       0.1137        0.1017          0.0540
         Lognormal, Variable      0.1130        0.1017          0.0566
         Normal, Uniform          0.1112        0.1017          0.0369
         Normal, Variable         0.1100        0.1017          0.0369

     **About 90 percent of it is shared.** The six methods' per-material errors
     correlate **0.66 to 0.99** and all six err in the SAME DIRECTION on **56.9
     percent** of materials, against about 3 percent if they were independent.
     That is exactly what a low NRMSE between methods beside a high error
     against the truth means, and decision 146 already records the pair; this
     names the mechanism.

     **THIRD, THE MECHANISM, AND IT IS THE ONE THIS PROJECT KEEPS FINDING.**

         n of the material   true ui   mean |error|   signed error
         3-9                  0.270       0.166          -0.093
         10-99                0.253       0.112          +0.005
         100-999              0.241       0.084          +0.042
         1000+                0.237       0.073          +0.046

     Every method understates the variance of a material estimated from three
     to nine values, because the spread of a distribution is not visible in nine
     points. A variance SHARE has to sum to one, so the share the small material
     loses is handed to the large ones -- which is why the signed error goes
     from -0.093 at the bottom to +0.046 at the top. **It is a property of the
     DATA, not of the method, which is why swapping methods does not fix it and
     why all six are wrong together.**

     **SO THE PAPER SAYS BOTH HALVES IN ONE SENTENCE.** Which material drives
     the uncertainty is the one question where the choice of UQ method barely
     matters, and it is also the question every method answers worst; a reader
     given only the first half would conclude the number is reliable.

160. **2026-09-23, Stage 2g review. "VARIABLE" IS RENAMED TO "DIRICHLET SHARES"
     FOR DISPLAY, because the old word carried a claim the method does not make
     -- and the oracle run shows the author's objection was right about
     weighting and wrong about what the column is.** `[AUTHOR]` "It still
     doesn't make any sense to me that lognormal with uniform weights would ever
     outperform lognormal with variable weights. To me, that's indicative of
     some mistake in our modeling process. It simply doesn't make sense that
     accounting for weights wouldn't improve the model."

     **THE PREMISE IS CORRECT AND THE MEASUREMENT CONFIRMS IT.** Mean absolute
     error in a material's estimated contribution against the market-weighted
     truth, 1,200 pLCA groups:

         family      equal    Dirichlet-drawn    the TRUE shares
         KDE         0.1285       0.1215             0.1064
         Lognormal   0.1243       0.1168             0.1007
         Normal      0.1620       0.1629             0.1550

         its chance of being largest
         KDE         0.0815       0.0818             0.0717
         Lognormal   0.0798       0.0847             0.0731
         Normal      0.1153       0.1187             0.1130

     **KNOWING THE SHARES BEATS EQUAL WEIGHTING ON EVERY METRIC AND EVERY
     FAMILY, without exception.** Accounting for weights does improve the model,
     exactly as the author says.

     **WHAT LOSES IS THE STAND-IN, NOT THE WEIGHTING.** Guessing the shares from
     a flat Dirichlet captures about a third of the available gain on the
     magnitude and LESS THAN NOTHING on the ranking: the lognormal's ranking
     error goes 0.0798 to 0.0847 when the shares are guessed, and would have
     gone to 0.0731 had they been known. The mechanism is the effective sample
     size -- a flat Dirichlet over n points leaves a Kish effective sample of
     about n/2, so guessing throws away half the data, and in this generator the
     within-mode split of a mode's share is uninformative by construction
     (decision 79).

     **THE SCORECARD DIFFERENCES ARE REAL, NOT NOISE**, which had to be checked
     before any of this could be said. Paired cluster bootstrap over pLCA
     groups, equal minus Dirichlet-drawn, negative meaning equal weighting is
     closer:

         its chance of being largest       -2.14  [-3.12, -1.35]
         its share of the total            -0.62  [-0.91, -0.29]
         the total's standard deviation    -1.62  [-2.27, -0.95]
         a material's mean contribution    +0.61  [+0.35, +0.91]
         its 95th percentile               +1.34  [+0.96, +1.73]
         its share at the building's 95th  +0.87  [+0.33, +1.43]

     Guessed shares help the LEVELS and hurt the RANKING, which is decision
     121 reaching the claim level.

     **SO THE WORD IS THE DEFECT.** "Variable" reads as "market shares accounted
     for". It means "market shares drawn from a flat Dirichlet because nobody
     publishes them". `fitting.WT_DISPLAY` maps Uniform to "equal weights",
     Variable to **"Dirichlet shares"** and the oracle scheme to "true shares".

     **WHY "DIRICHLET" AND NOT "GUESSED" OR "ASSUMED".** The Dirichlet is the
     instrument Torres, Lupton, Marsh, Srubar and Allen (2026) puts in its own
     title, so this names the companion paper's mechanism rather than inventing
     a third vocabulary, and it cannot be read as "the shares are known". A
     judgment word would also editorialize in a figure axis.

     **IT IS A DISPLAY LABEL AND THE DATA IS UNCHANGED.** The stored `method`
     values keep "Uniform" and "Variable" because they are the join key between
     every table this study writes and the eight regression fixtures that pin
     them; renaming the data would move numbers for a presentation fix.
     `tests/test_metricset.py` pins the mapping and pins that `fitting.PEWT` is
     untouched. **Stage 3 owns the figures and should carry the display labels
     into the rest of them**; Stage 2g applies them to its own figure only.

161. **2026-09-23, Stage 2g review. THE SCORECARD GAINS A SIZE-BAND PANEL,
     because counting boxes on the pooled figure reads as a verdict for the
     lognormal and the ordering INVERTS at about 100 declarations.** `[AUTHOR]`
     "Do you think this figure makes it look like lognormal is preferred to KDE?
     That makes me a bit worried, because I don't want people to take that away
     from this paper. But it has 10 categories it performs best in vs 5 for KDE.
     What exactly is our narrative here?"

     **THE WORRY IS JUSTIFIED AND THE MISREADING CUTS AGAINST THE KERNEL
     ESTIMATE.** Mean error across the seven per-material claims, each as a
     percentage of its own true level, by the material's own dataset size:

         n          KDE eq  KDE Dir  Logn eq  Logn Dir  Norm eq  Norm Dir
         3-9          40.4    43.6     40.1     45.5      41.8     45.6
         10-99        25.7    26.8     23.7     25.9      27.5     28.3
         100-999      18.1    15.8     18.1     15.3      22.1     20.5
         1000+        16.3    11.0     17.2     12.6      20.8     17.9

     Closest method by band: the lognormal under EQUAL weights at 3-9 and at
     10-99, the lognormal under DIRICHLET shares at 100-999, and the KERNEL
     ESTIMATE under Dirichlet shares above 1,000. Counting claim by claim
     instead of pooling them: 3-9 goes 4 to the equal-weighted lognormal, 2 to
     the equal-weighted normal and 1 to the equal-weighted kernel estimate;
     10-99 goes 6 to the equal-weighted lognormal; 100-999 goes 5 to the
     Dirichlet lognormal and 2 to the Dirichlet kernel estimate; and 1000+ goes
     4 to the Dirichlet kernel estimate and 3 to the Dirichlet lognormal.

     **BOTH AXES INVERT, not just the family.** Equal weights win every band
     below 100 declarations and Dirichlet shares win every band above, which is
     decision 140 from the claim side.

     **WHY THE POOLED COUNT FAVORS THE LOGNORMAL: it is an average over a size
     mix that is a design choice.** The corpus allocates 2,500 datasets to each
     of four size bands (decision 19), so half of every pLCA sits below 100
     declarations. **Reweighting to the real size mix of the 147 EC3 categories
     -- 14 / 54 / 26 / 6 percent -- moves it FURTHER toward the lognormal**, to
     three claims for the equal-weighted lognormal, because two thirds of real
     categories hold fewer than 100 declarations.

     **SO THE NARRATIVE IS NOT "USE THE KERNEL ESTIMATE", and this figure is not
     evidence for it.** What the paper can say: the normal is the one clear
     loser, and even that is conditional, because at three to nine declarations
     nothing can be estimated and a normal is as good as anything. Kernel
     estimate versus lognormal is a SIZE RULE and not a verdict, at a threshold
     of about 81 declarations (decision 142). And whether to weight is a size
     rule too. **A box count on a pooled figure is not a ranking of methods**,
     and before this review nothing on the figure said so.

     **THE FIX IS THE MECHANISM, NOT A WARNING.** A caption telling the reader
     not to count boxes would have asked them to take it on trust. The figure
     now carries a second panel, four size bands by six methods, on the same
     color scale because it is the same quantity in the same units, with the
     closest method in each band boxed exactly as in the panel above.
     `metricset.size_band_recovery`, `TABLE_MetricSizeBands.csv`.

162. **2026-09-23, Stage 2g review. THE DESIGN COMPARISON GOES FROM 800 PAIRS TO
     2,500, because 800 was a cost cap and the interval on the stage's headline
     null was as wide as the null.** `[AUTHOR]` "Did we only do 800 comparisons
     here? Seems like a small n, right? Why wouldn't we do more?"

     **THE CONCLUSION WAS NEVER AT RISK AND THE PRECISION WAS.** Bootstrapping
     over pairs at 800:

         B claimed to save   the truth        spread over the six methods
         0 pct               0.505 +/- 0.009  0.006   (97.5th pct 0.019)
         5 pct               0.629 +/- 0.010  0.012   (97.5th pct 0.023)
         10 pct              0.754 +/- 0.009  0.020   (97.5th pct 0.028)

     The spread never approaches anything a design decision turns on, so the
     null of decision 118 stands. But the interval on that spread is as wide as
     the spread itself, so "at most two percentage points" was really "at most
     about three".

     **2,500 IS THE NUMBER EVERY OTHER TRUTH-RUN RESULT IN THIS STUDY USES**, so
     it also removes an unexplained inconsistency: the design comparison was the
     one experiment run at a tenth of the sample. It tightens both intervals by
     a factor of 1.8 and costs a few minutes of the notebook run.

     **AND 2,500 IS ENOUGH; MORE BUYS NOTHING. Measured 2026-09-24 after the
     author asked about 5,000 and 10,000.** Subsampling the pairs on disk, the
     standard error of the spread across the six methods is 0.0046 at 312
     pairs, 0.0032 at 625, 0.0023 at 1,250 and **0.0012 at 2,500**, against a
     spread of **0.0152**. The effect is already thirteen standard errors clear
     of its own noise. Doubling to 5,000 would give 0.0009 and 10,000 would
     give 0.0006, on a quantity the paper reports as "at most about 1.5
     percentage points". **The run time would roughly double for a third
     decimal place nobody reads.**

     **AND THIS IS WHY 800 LOOKED WORSE THAN IT WAS.** The reported spread is a
     MAXIMUM over six methods, and a maximum over noisy estimates is biased
     upward when the estimates are noisy. The same subsampling shows it: the
     mean spread reads 0.0192 at 312 pairs, 0.0172 at 625, 0.0158 at 1,250 and
     0.0152 at 2,500. So part of what the larger sample bought was removing a
     bias in the statistic, not just narrowing an interval -- which is the same
     mechanism decision 108 records for the flip thresholds under a maximum.

163. **2026-09-23, Stage 2g review. THE "81 DECLARATIONS" THRESHOLD IS A
     GOODNESS-OF-FIT THRESHOLD AND THE CLAIM-LEVEL ONE IS NEAR 500. Quoting the
     first as though it settled the second is a conflation this project has
     been making, and it is why the kernel estimate does not sweep the
     scorecard.** `[AUTHOR]` "If KDE is better than lognormal at n > 81, why
     isn't KDE dominating most of these? After all, most datasets by a large
     majority are n > 81, right? Why does lognormal still look so good? That
     doesn't make any sense to me."

     **FOUR REASONS, AND THE PREMISE IS THE SMALLEST OF THEM.**

     **One, the size mix.** Above n = 81 is **52.1 percent** of the corpus, a
     coin flip rather than a large majority, because the corpus allocates 2,500
     datasets to each of four size bands. In the real EC3 arm it is **31.3
     percent** above n = 100.

     **Two, and this is the substantive one: the two thresholds measure
     different things.** Decision 142's 81 is where the kernel estimate
     overtakes the three-parameter lognormal on **W1 against the parent for one
     dataset**. The downstream claims cross much later. Mean error over the
     seven per-material claims, each as a pct of its own true level, under
     Dirichlet shares:

         n            KDE   lognormal   KDE minus lognormal
         3-9         43.6     45.5           -1.9   KDE ahead
         10-30       30.0     29.7           +0.3
         31-81       24.1     22.6           +1.5   lognormal ahead
         82-200      19.8     18.2           +1.6   lognormal ahead
         201-500     14.8     14.5           +0.3
         501-1000    13.3     13.7           -0.4   KDE ahead
         1001-3000   11.8     13.1           -1.3
         3000+       10.4     12.2           -1.9

     **NARROWED THE SAME DAY, BEFORE ANY OF IT WAS PUBLISHED. The first version
     of this entry called 500 "the claim-level crossover", as though it were a
     second law beside the 81. It is not, and the author's follow-up is what
     exposed it: "are you suggesting that stronger fit doesn't translate to
     agreeing on these claims?"**

     **FIT DOES TRANSLATE, and that had to be measured before anything else
     could be said.** Holding the material fixed and ranking the six methods by
     their fit and by their claim error: the median within-material Spearman is
     **+0.600**, it is positive on **82.2 percent** of materials, and the
     best-FITTING method is also the most claim-accurate on **39.0 percent** of
     materials against a **16.7 percent** chance level. It holds at every size,
     from +0.600 at n = 3-9 to +0.600 above 1,000. **So there is no finding here
     that a good fit fails to buy a good answer.**

     **WHAT THE 500 ACTUALLY IS: a group decision indexed on one member's
     size.** All four materials in a pLCA are fitted by the SAME method, so the
     group's composition decides which method wins the group, and binning by
     the focal material's own n asks a question the decision does not answer.
     The same 10,000 materials, binned instead by the GROUP'S MEDIAN n, put the
     kernel estimate ahead in **seven of eight bins** -- +1.9 at a group median
     of 3-9, +0.6 at 10-30, +0.1 at 31-81, -0.8 at 82-200, +0.6 at 201-500,
     +1.0 at 501-1000, +0.8 at 1001-3000 and +2.1 at 3000+, where positive
     means the kernel estimate is closer. Two partitions of the same rows,
     two different pictures.

     **SO THE CLAIM IS THE NARROW ONE AND 500 IS NOT A THRESHOLD TO QUOTE.** A
     threshold measured on ONE dataset's fit does not transfer unchanged to a
     decision made once for a GROUP of four, and the paper must say which
     criterion -- and which unit -- a threshold belongs to every time it quotes
     one. The kernel estimate lagging through the 82-200 band when binned on
     the focal material is the visible symptom of that mismatch, not evidence of
     a second crossover.

     **Three, a probabilistic LCA claim belongs to the GROUP of four, not to one
     material.** Groups are formed at random, so a large material almost always
     sits beside a small one and the group's worst member sets the group's
     error. Materials with n > 1000, split by the SMALLEST dataset in their own
     group:

         smallest in the group   materials   KDE   lognormal   KDE advantage
         3-9                       1,409     13.9     15.7          1.8
         10-99                       757      8.3      9.4          1.1
         100-999                     286      5.3      7.0          1.7
         1000+                        48      3.6      6.9          3.3

     A big material among big neighbors sits at **3.6 against 6.9** and the
     kernel estimate's edge nearly doubles. That configuration occurs **48 times
     in 10,000**. So the benefit of fitting one material well is diluted by
     whatever it is grouped with, and random grouping guarantees the dilution.

     **Four, where the kernel estimate wins is the SPREAD and not the LEVEL.**
     At n > 1000 under Dirichlet shares its advantage is +6.1 points on a
     material's standard deviation, +2.2 on its chance of being largest, +1.8
     on the uncertainty index and +1.4 on its 95th percentile -- and it **ties
     or slightly loses** on the mean contribution (1.6 against 1.5) and the mean
     share (5.6 against 5.6). Representing shape is what a kernel estimate buys;
     a mean is easy for every method.

     **SO "THE KERNEL ESTIMATE SHOULD DOMINATE" WAS NEVER WHAT THE EVIDENCE
     SAID**, and the scorecard is not hiding anything. The margins above the
     crossover are 1 to 2 points, which is a preference and not a dominance.

164. **2026-09-23, Stage 2g review. NOTEBOOK 3'S FIGURE CELLS ARE MARKED AND THE
     RENDERER ACCEPTS IT: a figure round is 1.6 seconds instead of a 48-minute
     run.** `[AUTHOR]` "Is this always going to take an hour between runs? That
     seems exhausting. Surely this figure doesn't take that long to produce,
     right? Can't we have tighter rounds of editing?"

     **THE FIGURE NEVER TOOK AN HOUR; THE TABLES UNDER IT DID.** The run is
     2,500 probabilistic LCAs by six methods by 10,000 draws, plus the sweeps,
     the run against the true parents and the design comparison.
     `audits/render_figures.py` has existed since Stage 2f to re-execute a
     notebook's own figure cells against the tables already on disk, and it
     **refused notebook 3** for two reasons, both now fixed.

     **One: eight figure-saving cells predated the `# FIGURE` convention** --
     cells 8, 29, 30, 35, 40, 49, 50 and 51, which draw figures the paper and
     the supplement use. The tool refuses rather than redrawing a subset
     silently (decision 56), so one unmarked cell disabled it for the whole
     notebook. They are marked, with the marker added above the cell and
     nothing else changed.

     **Two: the setup was taken to be the single cell that defines `OUT`.**
     Notebook 4 puts its imports and its output root in one cell; notebook 3
     keeps its imports two cells earlier, so executing only the `OUT` cell gave
     a NameError on numpy before the first figure drew. `read_cells` now returns
     every code cell up to and including the one defining `OUT`, **as a list
     executed one cell at a time in the notebook's own order**. A list and not a
     concatenation: a cell whose source has no trailing newline runs fine in a
     notebook and glues onto the next one when joined, which produced
     `generate_dontread = Falseimport sys` and a SyntaxError.

     **THE GUARANTEE IS UNCHANGED and is what makes the second executor safe.**
     The script still holds no figure code and no analysis code; it executes the
     notebook's own bytes. `tests/test_render_figures.py` pins that the sources
     executed are byte-identical to the notebook's and that notebook 3 is fully
     marked. **Verified end to end: rendering the scorecard alone took 1.6
     seconds and produced a PNG with the same sha256 as the one the 48-minute
     run wrote.**

         python audits/render_figures.py 03_CompareUQ_PerformPLCA \
             --only "every claim" --into-outputs

     **WHEN A FULL RUN IS STILL REQUIRED: whenever a TABLE changes.** A figure
     edit is seconds; a change to what is computed is a run. Of this stage's
     review, the denominator change and the design comparison's sample size
     needed runs and the size-band panel did not, because its table is derived
     from one already on disk -- and bundling the two cost the author an hour of
     waiting that better sequencing would have saved.

165. **2026-09-23, Stage 2g review. DECISION 56 IS RESTATED IN THE AUTHOR'S OWN
     TERMS: what matters is that the CODE for anything in `outputs/` lives in a
     notebook and can be reviewed there, not which process wrote the bytes.**
     `[AUTHOR]` "Maybe we need to restate the project rule that outputs are
     written by the notebooks and nothing else. The way that rule should read is
     actually something more like 'all figures in outputs/ should be
     reproducible in notebooks'. I just don't want you creating figures where
     the code is lost or not in a notebook. Notebooks are the best way to review
     code for figures."

     **THE OLD RULE NAMED THE WRONG THING.** "`outputs/` is written by the
     notebooks and by nothing else" constrains the WRITER. What the author cares
     about is that the code is reviewable in a notebook, and the writer was only
     ever a proxy for that. The proxy cost an hour of the author's time twice in
     one afternoon: re-slicing a table that was already on disk required
     re-running 2,500 probabilistic LCAs, not because the arithmetic needed it
     but because the rule said the notebook had to be the writer.

     **THE RULE AS IT NOW READS.** Every figure and every table in `outputs/`
     must be REPRODUCIBLE FROM A NOTEBOOK CELL, and the code that produces it
     must live in a notebook. Nothing may reach `outputs/` whose source is a
     scratch script, a chat session, or anything else a reviewer cannot open
     and read. A DIFFERENT EXECUTOR of the notebook's own bytes is permitted;
     a different AUTHOR of the bytes is not.

     **I HAVE GENERALIZED "figures" TO "figures and tables" and the author
     should overrule this if it is wider than intended.** The reasoning is that
     the author's stated concern -- code that is lost or not reviewable -- is
     identical for a table, and the cost being paid is mostly on tables: the
     scorecard and the size-band split are pure re-slices of tables already on
     disk, and the expensive part of that whole block takes **9 seconds** when
     driven from disk against 48 minutes for the run that regenerates what it
     reads.

     **WHAT DOES NOT CHANGE, AND IS THE REASON THE OLD RULE EXISTED.** Two
     figures dated 2026-03 once had no producer anywhere in the repository and
     had to be deleted; one figure had been written by a scratch script. That
     must not recur, and the restatement forbids it as squarely as the original
     did. The enforcement also stands: `audits/render_figures.py` contains no
     figure code and no analysis code, it executes the notebook's own cells
     verbatim, and `tests/test_render_figures.py` asserts the sources executed
     are byte-identical to the notebook's and that the module holds no plotting
     call. **A partial re-run must still refuse rather than skip**, because
     redrawing some outputs and reporting success leaves stale ones committed
     beside fresh ones with nothing to say so.

     **STILL OPEN, AND STAGE 3 OWNS IT:** extending the renderer from figure
     cells to table cells needs a rule for what a safe partial re-run of table
     cells IS, since a table computed from a stale input is a worse failure than
     a stale figure. The honest version is probably "execute every cell from the
     first one that writes a table", which is cheap here because that block
     reads from disk. Decision 56's narrowing is unchanged until that exists.

166. **2026-09-23, Stage 2g review. A GOODNESS-OF-FIT ADVANTAGE IS HEAVILY
     ATTENUATED BY THE TIME IT REACHES A pLCA ANSWER: the fit threshold is
     about 81 declarations and the claim threshold is about 1,000. The author
     was right and this entry's first version was wrong.** `[AUTHOR]`

     **WHAT THIS ENTRY FIRST SAID, AND WHY IT WAS WRONG.** It was headed
     "'goodness-of-fit doesn't really matter for actual pLCAs' is FALSE", on
     the strength of a within-material rank correlation of +0.600 between fit
     and claim error. That correlation is real and it says the DIRECTION
     transfers. It does not say the MAGNITUDE does, and the entry used it as
     though it did. Put side by side, equal weights:

         n           KDE wins fit   median fit gap   KDE wins claim   claim gap
         3-9             58.1          -2.3 pct          53.5          +0.3
         10-81           47.7          +0.8              48.4          +2.1
         82-200          61.2          -4.5              49.8          +0.5
         201-500         68.1          -9.1              54.7          +0.0
         501-1000        79.9         -15.3              58.0          -0.8
         1000+           85.5         -20.6              62.4          -0.9

     **At 201-500 the kernel estimate fits better on 68 percent of datasets by
     a median of 9 percent and buys ZERO claim accuracy.** The fit column
     crosses half at 82-200; the claim column does not cross until 1,000. That
     is an order of magnitude in the threshold a practitioner would act on, and
     it is a finding rather than an artifact.

     **WHAT SURVIVES FROM THE ORIGINAL ENTRY**, and it is the mechanism rather
     than a rebuttal: fit is not irrelevant -- the best-fitting method is the
     most claim-accurate on 39.0 percent of materials against a 16.7 percent
     chance level, and the ordering does follow at the top of the size range.
     The attenuation has a cause: **a probabilistic LCA chooses ONE method for
     all four materials in a group**, so a material's own fit advantage is
     averaged against three neighbors drawn at random from the whole corpus,
     most of them below the threshold where that advantage exists (decision
     163).

     **THE SENTENCE THE PAPER SHOULD CARRY:** a goodness-of-fit comparison
     ranks methods correctly but overstates how much choosing the better one
     buys, and a threshold calibrated on fit must not be quoted as the
     threshold for a probabilistic LCA -- here they differ by a factor of more
     than ten.

     **AND THE EXPERIMENT THIS OPENS, which no stage has run.** If the
     attenuation is the group averaging, then letting the method vary BY
     MATERIAL -- a kernel estimate on the well-populated categories, a
     three-parameter lognormal on the sparse ones, which is a policy a
     practitioner can actually follow -- should recover much of the fit
     advantage at the claim level. Every pLCA in this study uses one method for
     all four materials, so the mixed policy has never been measured
     downstream. It needs its own truth run and it is the most valuable
     experiment left. **Owner: whoever runs the next pLCA stage.**

     The original entry's text follows, superseded.

     **["GOODNESS-OF-FIT DOESN'T REALLY MATTER" IS FALSE -- SUPERSEDED ABOVE]**
     `[AUTHOR PROPOSED, MEASUREMENT DISAGREED, MEASUREMENT WAS TOO NARROW]` The
     author's proposed takeaway: "goodness-of-fit doesn't really matter for
     actual pLCAs, because otherwise KDE would dominate for n > ~80."

     **FIT TRANSLATES.** Holding the material fixed and ranking the six methods
     by their fit and by their claim error, the median within-material Spearman
     is **+0.600**, positive on **82.2 percent** of materials, and the
     best-fitting method is also the most claim-accurate on **39.0 percent**
     against a **16.7 percent** chance level -- at every dataset size.

     **THE PREMISE IS WHAT FAILS: the kernel estimate does not dominate the FIT
     above 80 either.** Share of datasets where it beats the three-parameter
     lognormal on W1 against the parent, and the median relative gap:

         n            wins, equal   wins, sampled   median gap (sampled)
         3-9             58.1           70.7              -5.2 pct
         10-30           46.2           56.7              -2.7
         31-81           49.6           46.3              +1.5
         82-200          61.2           53.0              -2.1
         201-500         68.1           66.9             -13.2
         501-1000        79.9           79.8             -28.6
         1001-3000       84.4           88.2             -48.5
         3000+           86.5           95.7             -67.5

     **81 is where the kernel estimate crosses 50 percent, not where it
     dominates.** At 82-200 it wins barely half the time by 2 percent; only
     above 500 does it win four times in five by 13 to 68 percent. **The
     claim-level picture tracks that faithfully** -- barely ahead where the fit
     is barely ahead, clearly ahead where the fit is clearly ahead -- which is
     what a Spearman of +0.600 predicts. Nothing here says fit is irrelevant.

167. **2026-09-23, Stage 2g review. THE LOGNORMAL THIS STUDY FITS IS NOT THE
     LOGNORMAL THE FIELD USES, and against the one the field uses the kernel
     estimate wins by 31 to 41 percent. The manuscript must draw this
     comparison explicitly.** `[AUTHOR]` "What machinery did this paper have to
     invent for lognormal distribution fitting? I think the comparison between
     our lognormal method and a regular two-parameter lognormal should be made
     very clear in this manuscript."

     **THE MACHINERY, and it is not cosmetic.** A three-parameter lognormal has
     no global maximum likelihood estimate: the likelihood is unbounded as the
     threshold approaches the smallest observation. So this study had to build
     (a) `families.fit_lognorm3_profile`, which chooses the threshold by PROFILE
     LIKELIHOOD over a grid; (b) the guard `PROFILE_DELTA_LO_FRAC = 0.25`,
     calibrated on a BOUNDED-VARIANCE criterion -- the smallest guard at which
     no fitted model on either arm exceeds five times the data's own standard
     deviation -- deliberately NOT on W1, so it is not tuned to the score it is
     judged by, and which **determines the threshold for 48 percent of
     empirical fits**; (c) an explicit truncation to (0, inf) with
     renormalization, so the object scored is the object sampled (decision 50);
     and (d) the analytic tail term, without which the lognormal alone is
     under-charged for mass beyond the scoring grid (decision 85). At a guard of
     0.01 instead of 0.25 the same estimator produced a model with a standard
     deviation of 3,281 on data whose own is 0.6 (decision 51).

     **AGAINST THE KNOWN PARENT, 1,500 corpus datasets, equal weights**,
     `audits/lognormal_variants.py`, median W1:

         n           2-param   3-param   gamma   normal     KDE
         3-9          0.2449    0.2189  0.2241   0.2567  0.2464
         10-99        0.1385    0.1123  0.1236   0.1694  0.1238
         100-999      0.1232    0.0777  0.0940   0.1842  0.0686
         1000+        0.1229    0.0750  0.0908   0.1809  0.0559

     **Closest on: the KERNEL ESTIMATE 47.8 percent of datasets, the
     three-parameter lognormal 17.8, the two-parameter lognormal 14.7, the
     normal 10.5, gamma 9.3.** The kernel estimate is the plurality winner by
     nearly three to one over its nearest rival.

     **MEDIAN RELATIVE GAIN OVER THE TWO-PARAMETER LOGNORMAL**, which is
     ecoinvent's default and what the pedigree matrix produces, since a
     geometric standard deviation IS a lognormal parameterization:

         n           3-param    KDE
         3-9           +0.1    -2.9
         10-99         -2.9    -5.8
         100-999      -11.5   -30.8
         1000+        -14.8   -41.2

     The kernel estimate beats it on **71.5 percent** of datasets and the
     three-parameter lognormal on 63.3.

     **SO "THE LOGNORMAL WAS RIGHT ALL ALONG" IS NOT WHAT THIS SHOWS.** The
     lognormal that competes with the kernel estimate is a three-parameter fit
     with a profile-likelihood threshold and a calibrated guard, which is not
     what practitioners fit and which this project had to construct. Against
     what practitioners DO fit, the kernel estimate is 31 to 41 percent closer
     to the truth above 100 declarations. **The paper owes a direct
     two-parameter comparison in the results, not a footnote**, because without
     it a reader will assume the study's lognormal is theirs.

     **The honest tension that survives**: on the pLCA CLAIMS the kernel
     estimate and the three-parameter lognormal are a near tie below about 500
     declarations, while on FIT the kernel estimate is a clear plurality
     winner. Decision 163 explains the gap -- a claim is decided once for a
     group of four -- and decision 166 shows it is not because fit fails to
     translate.

     **THE FAIR-COMPARISON CONTROL, added 2026-09-23 from
     `audits/family_comparison.py` at 1,200 datasets, because a reviewer will
     ask it.** Every parametric family in this study is fitted by maximum
     likelihood and judged by W1, which are different criteria, so a family can
     lose for having been fitted under the wrong rule. Refitting each one by
     direct W1 minimization -- which no practitioner does, since it means
     minimizing the very distance you will then report -- the two-parameter
     lognormal improves by a median of **17.4 percent** and the normal by 18.2.
     **It does not overturn the comparison.** In-sample W1, equal weights,
     share of datasets on which the maximum-likelihood KERNEL ESTIMATE is still
     closer:

         two-parameter lognormal, fitted by MLE          75.6 pct
         two-parameter lognormal, fitted W1-optimally    67.7 pct
         three-parameter lognormal, fitted by MLE        64.4 pct
         three-parameter lognormal, fitted W1-optimally  54.4 pct

     So against the fit a practitioner would actually produce the kernel
     estimate wins three times in four, and against a two-parameter lognormal
     tuned to the scoreboard it still wins two times in three. The
     three-parameter lognormal tuned the same way is the only one that reaches
     a coin flip, which is the same near-tie the claim scorecard shows. **These
     are IN-SAMPLE and therefore flatter the flexible model**, so read them as
     an upper bound for the kernel estimate and the out-of-sample figures above
     -- 71.5 percent against the two-parameter fit, scored against the known
     parent -- as the ones to quote.

168. **2026-09-23, Stage 2g review. THE CORPUS'S MULTIMODAL DATASETS ARE THE
     WRONG SHAPE, NOT MERELY TOO FEW, AND DECISION 82'S REASSURANCE DOES NOT
     ANSWER IT. The author was right to reopen this.** `[AUTHOR ASKED,
     MEASUREMENT AGREES, REGENERATION IS THE AUTHOR'S CALL]` "Why does the
     corpus under-represent strongly multimodal datasets??? That's one of the
     main purposes of the corpus ... I feel fairly strongly that we should go
     back to data generation."

     **WHY DECISION 82'S TEST DOES NOT SETTLE IT.** That decision reweighted the
     corpus to the empirical mode mix, found the kernel-estimate-minus-lognormal
     difference moved by 0.0004, and closed the question. **Reweighting can only
     reweight datasets that exist.** If the corpus's multimodal datasets are a
     different object from real ones, no weighting of them reproduces the real
     population, and the 0.0004 measures the wrong thing. That is the case here.

     **THE SIGN IS OPPOSITE ON ALL SIX CHARACTERISTICS.** Spearman of the
     visible mode count with each characteristic, `audits/corpus_modality_shape.py`:

         characteristic    empirical   synthetic
         coeffvar            +0.163      -0.153
         skewness            +0.211      -0.152
         kurtosis            +0.230      -0.144
         fit_lognorm_SF      +0.086      -0.009
         fit_norm_SF         -0.202      +0.128
         crit_bw_1           +0.283      -0.115

     **In the real world more modes come WITH more spread, more skew and more
     kurtosis. In the corpus they come with LESS of all three.** It survives
     conditioning: inside the window 0.4 < CV < 1.2 the corpus still gives
     -0.102 against the real arm's +0.157. Medians by mode count make it vivid --
     a three-mode real category has a coefficient of variation of 0.974 and a
     three-mode synthetic one has **0.230**, which is TIGHTER than the corpus's
     own unimodal median of 0.593.

     **THE MECHANISM, which is the generator's and is fixable.** The corpus
     makes a second visible mode by SEPARATING components: median achieved
     overlap falls 0.513, 0.422, 0.411 as the visible mode count goes 1, 2, 3,
     and Spearman(overlap, modes) is -0.234. Separated components are each
     individually tidy, so a multi-mode synthetic dataset is LESS skewed and
     LESS dispersed than a unimodal one. Real multimodality is a shoulder on a
     long-tailed body. **Decision 37 predicted exactly this in Stage 2a-2** --
     "their modes are gentle shoulders on a lognormal body; the generator
     reaches the same mode COUNT by separating components, which is a different
     shape" -- and it was never measured against the mode count, so it never
     reached a decision.

     **WHY THE TUNING NEVER CAUGHT IT.** The calibration objective matches
     MARGINAL distributions one characteristic at a time (decision 35). Nothing
     in it looks at the CORRELATION between characteristics, so the generator
     can match every margin and still get the joint structure backwards.

     **THE SIZE OF THE HOLE.** Of the corpus's 1,718 multimodal datasets, only
     **8.0 percent** reach the real multimodal median coefficient of variation
     of 0.889, **21.5 percent** the real median skewness of 2.274 and **25.0
     percent** the real median kurtosis of 7.808 -- against 50 percent if the
     arms matched. Taken jointly, multimodal AND dispersed is **1.9 percent of
     the corpus against 16.2 percent of the real arm, an 8.4-fold
     under-representation** of a sixth of the real categories.

     **AND THE DIRECTION OF THE BIAS IS NOT ESTABLISHED. THIS MATTERS, because
     regeneration might not rescue the kernel estimate.** On the corpus's own
     (mis-shaped) multimodal-and-dispersed cell the kernel estimate does WORSE,
     not better, by 4.7 percent under equal weights on 153 datasets. On the real
     arm's 21 multimodal-and-dispersed categories it is a coin flip to worse:
     47.6 percent win share under equal weights, 38.1 under sampled shares. **So
     the case for regenerating is that the corpus is nearly silent about a sixth
     of real categories, not that it is hiding a kernel-estimate win.**

     **AND A FINDING THAT IS THE OPPOSITE OF THE USUAL INTUITION:** on the real
     arm the kernel estimate loses worst on UNIMODAL AND DISPERSED categories --
     26 of them, win share **15.4 percent** under equal weights and **3.8
     percent** under sampled shares, median 18 to 24 percent worse. Irregularity
     in the sense of a long tail is where a kernel estimate struggles; the
     multi-humped case is not its problem.

     **WHAT REGENERATION WOULD HAVE TO CHANGE**, and it is not "more modes": the
     generator must produce a second mode as a SHOULDER on a skewed body rather
     than as a separated tidy hump, which means coupling the modality target to
     the skewness and dispersion targets instead of drawing them independently,
     and adding a joint term -- the modality-dispersion correlation -- to the
     calibration objective so the failure cannot recur silently.

     **NOTHING IS REGENERATED HERE.** Generation is closed by decisions 47, 48
     and 55, every number in the paper moves when it reopens, and the cost is
     the full verification chain. The diagnosis, the audits and the proposed fix
     are recorded; **the decision is the author's.** If it goes ahead, the first
     step is a 1,000-dataset DRAFT corpus under decision 41, never a paper
     number, to check that the coupled generator fixes the joint structure
     without wrecking the margins the current one matches well.

169. **2026-09-23, Stage 2g review. SEVEN CANDIDATE CONFIGURATIONS AT 1,000
     DATASETS EACH, AND NOT ONE FIXES THE MODALITY-SHAPE SIGN. The defect is
     structural in how the generator builds a mode, so a regeneration with any
     of these settings would NOT fix what the author is asking about.**
     `[AUTHOR ASKED FOR THE TEST RUN, MEASUREMENT IS A CLEAN NEGATIVE]` "Let's
     do a little mini test run with 1000 datasets or something like that to see
     if multimodality is better."

     **THE TARGET.** Real ECC data: Spearman of the visible mode count with
     coeffvar +0.163, skewness +0.211, kurtosis +0.230, `fit_norm_SF` -0.202,
     `crit_bw_1` +0.283. The corpus has the opposite sign on all five.

     **THE CANDIDATES AND WHAT THEY DID.** `audits/corpus_joint_structure.py`,
     1,000 datasets each, drafts only:

         candidate                  multimodal  multi+dispersed  signs  objective
         current                      21.5 pct      1.27 pct     0 / 5    0.2506
         wider_dispersion             22.5          3.92         0 / 5    0.2909
         wider_and_blended            21.4          3.29         0 / 5    0.2808
         widest                       20.8          7.09         0 / 5    0.3711
         unequal_modes                24.4          1.77         0 / 5    0.3024
         unequal_and_wider            27.7          4.05         0 / 5    0.3363
         unequal_wider_separated      29.7          4.43         0 / 5    0.3702

         the real arm                 31.5 pct     16.2 pct         --        --

     **THE MODE COUNT IS REACHABLE AND THE SHAPE IS NOT.** Dropping
     `mode_share_alpha` from 10 to 1 -- the author's own proposal, carried to
     Stage 2h by decision 141 -- plus a wider dispersion ceiling and more
     separation gets the multimodal share to **29.7 percent against the real
     31.5**, which is a match. The multimodal-AND-dispersed share reaches 7.09
     percent at best against the real 16.2. **And the sign stays wrong on all
     five characteristics in every candidate**, between -0.096 and -0.251 on
     the coefficient of variation where the real arm is +0.163.

     **WHY NO PARAMETER REACHES IT.** The generator makes a visible mode by
     placing components apart and solving a spread multiplier for a target
     overlap. Whatever the mode SIZES and whatever the dispersion ceiling, a
     dataset built that way is a blend of separated bodies, and a blend of
     separated bodies is more symmetric than any one of them. Real
     multimodality is a small shoulder sitting ON the tail of a big skewed
     body. Those are different constructions and no setting of the current one
     produces the other.

     **AND THE MARGINS PAY FOR EVERY ATTEMPT.** The calibration objective goes
     from 0.2506 to between 0.2808 and 0.3711, and the paper's central quantity
     -- the uniform-to-variable Wasserstein distance -- degrades worst in the
     candidates that help modality most: 0.2295 at current, **0.9176** under
     `unequal_and_wider`, 0.8929 under `unequal_wider_separated`. That is
     decision 138's finding reproduced on a different lever.

     **SO THE RECOMMENDATION IS: DO NOT REGENERATE ON A PARAMETER CHANGE.** It
     would move every number in the paper, worsen the margins that currently
     match, and leave the joint structure exactly as wrong. If the corpus is to
     be fixed it needs a GENERATOR CHANGE -- secondary components drawn ON the
     primary's tail with smaller scale, rather than positioned independently
     and separated by an overlap solve -- plus a joint term in the calibration
     objective so a future configuration cannot match every margin while
     getting the correlation backwards. That is a redesign with its own
     verification, and it is the author's call whether this paper carries it or
     states the limitation.

     **WHAT TO STATE IF IT IS NOT FIXED.** The corpus spans the modality of
     real categories and the dispersion of real categories, and does not span
     their JOINT distribution: a synthetic multimodal dataset is tidier than a
     synthetic unimodal one, where a real multimodal category is wilder than a
     real unimodal one. Multimodal-and-dispersed is 1.9 percent of the corpus
     against 16.2 percent of the real arm.

     **AND AN INSTRUMENT FOR ANY FUTURE ATTEMPT.**
     `audits/corpus_examples.py` draws notebook 1's example-dataset panels for
     any candidate configuration, real categories in the bottom row, so a
     configuration can be judged by eye before it is judged by an objective.
     Written because the author asked to sanity-check the datasets being
     produced, and because the numbers above do not show what a shoulder looks
     like.

170. **2026-09-23 overnight, Stage 2g review. THE ARCHITECTURAL FIX WAS BUILT
     AND IT WORKS, AND IT SHOULD NOT BE ADOPTED: it improves the joint
     modality-dispersion structure tenfold and degrades the paper's CENTRAL
     QUANTITY twelvefold.** `[AUTHOR ASKED FOR IT BUILT AND TESTED, NOT
     ADOPTED]` "Build and test it ... I will NOT regenerate the production
     corpus or touch any paper number."

     **WHAT WAS BUILT.** `genconfig.separation_dispersion_frac`. For that share
     of multi-component parents, the component SPACING is solved so the mixture
     itself has the drawn coefficient of variation at its positivity floor, and
     the shift then does only what admissibility requires. Dispersion arrives
     WITH separation instead of in spite of it. The default is 0.0, no
     randomness is consumed there, and `tests/test_determinism.py` confirms the
     shipped generator is bit-identical.

     **THE SOLVE IS A SCAN AND THAT IS MEASURED, NOT ASSUMED.** The coefficient
     of variation is NOT monotone in the spread multiplier: on one drawn parent
     it runs 0.90, 0.88, 0.84, 0.79, 0.76, 0.89, 1.18, 1.46 as the multiplier
     goes 0 to 60, and on another 0.87 down to 0.28. Separating components
     widens the mixture and raises the floor it must clear, and which effect
     wins depends on the component shapes. A bisection would converge to
     whichever side it started from.

     **IT WORKS, ON THE REAL STRATIFIED DESIGN AT 1,000 DATASETS:**

         candidate           multi   disp   multi+disp   multi|disp   weighting
         current             0.215  0.135     0.0127       0.094        0.2295
         widest              0.208  0.479     0.0709       0.148        0.7191
         separation          0.281  0.410     0.0949       0.232        1.9649
         separation_kmin2    0.330  0.471     0.1266       0.269        2.7323
         the real arm        0.315  0.362     0.1620       0.447            --

     Multimodal-and-dispersed goes from **1.27 percent to 12.66**, a tenfold
     improvement that lands just short of the author's 16 percent target, and
     the multimodal share lands at 0.330 against the real 0.315.

     **AND THE PRICE IS THAT THE CORPUS WOULD OVERSTATE THE PAPER'S HEADLINE
     EFFECT BY A FACTOR OF 2.7. The first version of this entry said "the
     weighting quantity gets twelve times worse", which was a badly chosen
     phrase and the author caught it:** no VALUE of `w_v_uw_wasserstein` is
     better or worse than another -- it simply measures how far variable
     weighting moves a dataset. What matters is whether the corpus's
     DISTRIBUTION of it matches the real arm's, and under the separation path
     it does not:

         arm                        median    mean      max
         REAL, 147 categories       0.0929  0.1120   0.7302
         corpus, as shipped         0.1035  0.1394   1.0109
         corpus, separation         0.1966  0.3130   4.0641
         corpus, separation_kmin2   0.2513  0.3788  13.0823

     The corpus as shipped puts the median weighting effect at 0.1035 against
     a real 0.0929, which is close. The separation path puts it at **0.2513,
     2.7 times the real median**, with a maximum of 13.08 against a real
     maximum of 0.73. The standardized arm-to-arm distance on that
     characteristic, which is what the calibration objective sees, goes from
     **0.2915 to 2.5732**, and the overall objective from 0.2506 to 0.5619.

     **Since the paper's headline claim is HOW MUCH WEIGHTING MATTERS, a corpus
     that overstates it 2.7-fold would inflate the central result.** That is
     the reason not to adopt it, and it is a statement about calibration rather
     than about any value being good or bad.

     **THIS IS THE THIRD TIME THE SAME TRADE HAS APPEARED** -- decision 39 in
     Stage 2a-2, decision 138 in Stage 2f, and now here on a different lever
     each time. It is structural: in this generator, whatever widens the
     dispersion also widens the gap between the two weightings.

     **AND THE SIGN NEVER FLIPS. 36 CONFIGURATIONS**, 13 draft corpora and 23
     fast probes, and the modality-shape correlation is negative in every one:
     the best is -0.083 on the coefficient of variation against a real +0.163.
     Two constructions were tried and failed outright -- a shoulder that pairs
     the heaviest weight with the widest component (made multimodality worse,
     19.4 percent against 21.5, because a small component on a wide body's tail
     is buried in it) and its narrow-body mirror.

     **RECOMMENDATION: DO NOT REGENERATE. STATE THE LIMITATION.** The corpus
     spans the modality of real categories and the dispersion of real
     categories and does not span their JOINT distribution. Conditional on
     being dispersed, a real category is multimodal 44.7 percent of the time
     and a synthetic one 9.4. What the paper cannot speak to is the sixth of
     real categories that are both, and that bound should be stated in the
     limitations rather than engineered away at the cost of the weighting
     result. **The code is committed, defaulted off, and tested, so a later
     stage can revisit it if the weighting quantity ever stops being central.**

     **WHAT WOULD ACTUALLY BE NEEDED**, if a future stage wants both: the
     weighting effect and the dispersion are coupled through the mode-level
     market shares, so breaking the trade means changing the WEIGHT model at
     the same time as the shape model -- which is Stage 2h's first item
     (decision 141) and should not be attempted separately from it.

171. **2026-09-24, Stage 2g review. FIVE OF THE SIXTEEN SCORECARD ROWS AVERAGE
     SIGNED ERRORS BEFORE TAKING THE ABSOLUTE VALUE, AND THE OTHER ELEVEN DO
     NOT. The design comparison's "right to 0.8 percent" is an error in the
     AVERAGE over 2,500 comparisons, not the error on the one comparison a
     designer makes.** `[DELEGATED, found while answering the manuscript
     session's question about where the methods separate]`

     **WHAT DIFFERS.** The eleven magnitude and attribution rows read tables
     that take `|error|` per pLCA group or per material and then average --
     `total_mean__error_absmean`, and `recovery_table`'s `abs_error`. The four
     action rows and the comparison row read SUMMARY tables whose error column
     was already averaged over groups, so signed errors cancel before the
     absolute value is taken. Measured, as a percentage of each claim's true
     level:

         claim                          scorecard   per-group   best method
         a cap: how often it binds        0.48        33.01      per-group
         a cap: its mean saving           0.56        38.29
         a cap: its chance of saving 5    1.09        32.97
         using 25 pct less: mean saving   0.00        10.02
         the probability B beats A        0.81        11.95

     **THE MOST MISLEADING ONE IS THE QUANTITY REDUCTION.** The scorecard says
     every method is EXACTLY right on it, to four decimal places, and decision
     119 says the same. That is true of the average over 2,500 buildings. On a
     single building every method is **10 to 13 percent** out.

     **AND IT QUALIFIES THE STAGE'S HEADLINE NULL.** Decision 118 reports that
     the choice of method moves the stated probability that a substitution is
     an improvement by at most 0.015. That is the spread of the AVERAGE. On a
     single comparison the six methods span a median of:

         B claims to save    0 pct   1 pct   2 pct   5 pct  10 pct  20 pct
         median spread       0.182   0.182   0.180   0.165   0.119   0.025
         disagree on WHICH
           design is better  93 pct  80 pct  64 pct  28 pct  5.9 pct 0.04 pct

     The 0 percent column is the control -- the truth is a coin flip there, so
     disagreement means nothing. **What the table says is that the null holds
     where the design difference is real and fails where it is small: at a
     claimed 5 percent saving the six methods disagree about which design is
     better on 28 percent of individual comparisons, while their AVERAGE
     probabilities sit within 0.009 of each other.**

     **SO DECISION 118 IS NOT WRONG AND IS NOT THE WHOLE STATEMENT.** "The
     choice of UQ method is safe for a design comparison" is true of a
     practice averaged over many comparisons and false of one comparison near a
     tie, which is the same fragility this project records for the rank metric
     (decision 95, 101, 107) arriving at the design question.

     **NOT FIXED HERE, BECAUSE IT CHANGES THE STAGE'S HEADLINE FIGURE AND IS
     THE AUTHOR'S CALL.** Making all sixteen rows per-unit would move the
     figure's title from "right to 0.8 pct on the design comparison" to "right
     to 12.0 pct" and would raise the four action rows by 10 to 38 points. The
     alternative is to keep the averaged form and SAY on the figure that these
     five rows are errors in an average. Either is defensible; mixing them
     without saying so is not.

172. **2026-09-24, Stage 2g review. THE SECOND RULE OF THE SAFE-LEAD KIND: a
     design comparison is safe once the claimed saving exceeds about 10
     percent, and the number a practitioner needs is how often a method names
     the WRONG DESIGN, not how often two methods disagree.** `[MANUSCRIPT
     SESSION ASKED]`

     **THE PRACTITIONER'S NUMBER, 2,500 pairs from the production run, scored
     against the truth run on the same draws.** Share of individual comparisons
     on which a method lands on the opposite side of a half from the truth:

         B claims   truth      KDE     KDE      Logn     Logn    Norm    Norm
         to save   P(B better) equal  sampled   equal   sampled  equal  sampled
           0 pct      0.50      32.8    28.2     31.4     27.7    65.8    46.9
           1 pct      0.53      31.2    27.1     30.0     26.8    57.3    44.0
           2 pct      0.55      28.6    25.3     27.6     25.4    45.0    39.0
           5 pct      0.63      16.0    16.8     16.3     17.3    19.0    22.6
          10 pct      0.75       3.8     5.0      4.0      5.6     4.0     5.6
          20 pct      0.95       0.0     0.0      0.0      0.0     0.0     0.0

     **AT A CLAIMED 5 PERCENT SAVING EVERY METHOD NAMES THE WRONG DESIGN ABOUT
     ONE TIME IN SIX**, and the six sit within 6 points of each other, so the
     error rate is a property of the question rather than of the method. By 10
     percent it is 4 to 6 percent and by 20 percent it is zero.

     **AND THE NORMAL IS WORSE THAN A COIN TOSS WHERE THE TRUTH IS A COIN
     TOSS.** At a claimed 0 percent saving `Normal, Uniform` is wrong **65.8
     percent** of the time. That is not noise: the normal is systematically
     optimistic (decision 119), so when the truth sits at a half a systematic
     lean lands on the wrong side MORE often than chance. It converges to the
     others by 10 percent, which is the same pattern as everywhere else --
     the normal fails where the decision is close.

     **THE CROSSINGS, from a dense sweep because the production run could not
     locate them.** The study runs six savings -- 0, 1, 2, 5, 10, 20 percent --
     which leaves three of the four crossings inside a TEN-point gap with no
     observation between. `audits/swap_saving_threshold.py` fills 6, 7, 8, 9,
     12, 14, 16 and 18 percent on 900 pairs, so every crossing is bracketed
     within two points. Cluster bootstrap over design pairs:

         risk level                              claimed saving at which it crosses
         a method names the wrong design,  5 pct   10.15 pct  [9.57, 10.73]
         a method names the wrong design,  1 pct   14.57 pct  [13.46, 15.21]
         two methods disagree,             5 pct   11.10 pct  [10.42, 11.68]
         two methods disagree,             1 pct   15.20 pct  [14.12, 15.81]

     **THE RULE TO PRINT: a substitution claimed to save more than about 10
     percent of the building will be called correctly by any of these six
     methods 95 times in 100, and one claimed to save more than about 15
     percent 99 times in 100. Below 5 percent no method is reliable and the
     choice between them does not help.** That sits beside the safe-lead ratio
     of 2.13 (decision 107) as the second rule of its kind, and it is measured
     the same way.

     **A LOGISTIC FIT IN LOG SAVING DOES NOT WORK HERE and was discarded.** It
     is what decisions 95 and 107 used, and on this curve it puts the 5 percent
     crossing at 19.7 percent saving when the observed rate is already 4.68
     percent at 10. The decline is far steeper than a logistic, so the
     crossings are interpolated between bracketing observations instead, which
     the dense sweep makes safe.

     **THE SWEEP AGREES WITH THE PRODUCTION RUN where they overlap** -- at a
     claimed 5 percent saving the wrong-design rate is 18.01 percent on 2,500
     production pairs and 19.76 on the sweep's 900, and at 10 percent 4.68
     against 5.20 -- which is the check that the sweep is the same experiment
     at finer resolution rather than a different one.

173. **2026-09-24, Stage 2g review. ALL FOUR PUBLISHED CONSTANTS ARE BRACKETED
     BY OBSERVATIONS AND NONE IS AN EXTRAPOLATION -- but their published
     intervals are narrower than the disagreement between two link functions,
     so they understate uncertainty.** `[MANUSCRIPT SESSION ASKED]` The
     question, prompted by the logistic misfitting the design comparison: were
     the flip thresholds of 0.0018, 0.011 and 0.025 and the safe-lead ratio of **[SUPERSEDED CONSTANTS -- the published values are 0.0029, 0.015 and 0.032; see decision 223.]**
    
     2.13 produced by the same fit, and do observations sit either side of
     them?

     **YES TO THE SAME FIT. `flip.logistic_fit` in log space, with
     `flip.bootstrap_crossings` resampling clusters and refitting**, is what
     produced all four. `logistic_crossing` then solves
     `exp((logit(level) - a) / b)`, which is the same form that failed on the
     design comparison.

     **AND YES TO BRACKETING, WHICH IS WHY THEY DO NOT CARRY THE SAME ERROR.**
     Observed flip rates by binned separation, 22,500 calibration rows:

         separation (median)   0.0000  0.0024  0.0042  0.0064  0.0094  0.0136  0.0194  0.0274
         flip rate, pct          0.19    1.37    1.80    2.86    4.17    5.48    7.03   11.01

     The 1 percent level is bracketed between 0.0000 and 0.0024, the 5 percent
     level between 0.0094 and 0.0136, the 10 percent level between 0.0194 and
     0.0274 -- and the published 0.0018, 0.011 and 0.025 each sit INSIDE their
     bracket. The safe-lead ratio likewise, over 72,000 four-material
     comparisons:

         top-two ratio (median)  1.348  1.591  2.000  2.105  2.880
         flip rate, pct          13.16   5.27   2.96   1.20   0.22

     10 percent is bracketed between 1.348 and 1.591 and the published 1.46 is
     inside; 5 percent between 1.591 and 2.000 and 1.64 is inside; 1 percent
     between 2.105 and 2.880 and 2.13 is inside.

     **WHY THE LOGISTIC FAILED ON THE DESIGN COMPARISON AND NOT HERE.** The
     design comparison's predictor is a DESIGNED variable with six levels and a
     ten-point gap across the crossing; these two have 22,500 and 72,000
     observations spread continuously over the predictor. A two-parameter shape
     interpolating between dense observations is doing little work; the same
     shape spanning a 2x gap is doing all of it.

     **WHAT IS NONETHELESS WRONG WITH QUOTING THEM TO THREE FIGURES.** The
     published intervals are bootstrap intervals on the FITTED PARAMETER, and
     they are narrower than the disagreement between the logistic and the
     isotonic fit, which assumes only monotonicity:

         constant            logistic   isotonic   published interval
         safe lead, 1 pct      2.13       2.22       2.09 to 2.17
         safe lead, 5 pct      1.64       1.61       1.62 to 1.66
         safe lead, 10 pct     1.46       1.35       1.45 to 1.47
         flip, 1 pct          0.0018     0.0026     0.0013 to 0.0023
         flip, 5 pct          0.011      0.0129     0.0091 to 0.0126
         flip, 10 pct         0.025      0.0271     0.0217 to 0.0277

     **On four of the six the isotonic value sits OUTSIDE the published
     interval.** The interval says how well the data pin down a logistic's
     parameters; it says nothing about whether a logistic is the right shape.
     **Decision 95's instruction to quote TWO significant figures and no more
     already covers this, and it should be extended to the safe-lead ratio,
     which is currently quoted as 2.13.** At two figures the two fits agree on
     every one of the six except the 10 percent safe lead, where 1.5 against
     1.4 is the honest spread.

     **NOTHING NEEDS RE-DERIVING.** The constants stand; what needs changing is
     the precision they are printed to and, where the two fits differ by more
     than the last quoted figure, saying so. `audits/` holds the binned tables
     behind this entry; the check is one groupby and is worth repeating if the
     calibration is ever rerun.

174. **2026-09-24, Stage 2h. FIVE OF THE SIXTEEN SCORECARD ROWS WERE AVERAGING
     THE SIGNED ERROR BEFORE TAKING THE ABSOLUTE VALUE, so they reported a
     cancellation rather than an error. Every row is now the PER-UNIT error and
     both statistics are kept. THIS IMPLEMENTS DECISION 171**, which diagnosed
     it and left the choice to the author. `[AUTHOR]` "The author has decided
     to take the correction."

     Stage 2g's decision 157 put every scorecard row on one DIVISOR, the mean
     true LEVEL. This puts every row on one NUMERATOR. The four
     reduction-strategy rows and the design comparison read summary tables that
     averaged the signed error over pLCA groups first, so a method too high on
     one building and too low on the next reported almost nothing; the other
     eleven rows were already per unit. The figure was again drawing two
     statistics on one color scale.

     **WHAT MOVED, best method, as a percentage of each claim's own true
     level:**

         a cap: how often it binds            0.48 -> 30.62
         a cap: its mean saving               0.56 -> 38.29
         a cap: its chance of saving 5 pct    1.09 -> 32.97
         using 25 pct less: its mean saving   0.00 -> 10.02
         the probability B beats A            0.81 -> 11.95

     **The eleven other rows are bit-identical**, and the new `portfolio_error`
     column reproduces the old values to 2e-15, which is the check that this is
     a change of DEFINITION and not of computation.

     **BOTH ARE KEPT AND BOTH ARE LABELED**, because the averaged form is a
     real quantity a different reader wants rather than a worse version of the
     first: `total_error` is the error in a SINGLE decision, which is what a
     designer choosing between two options carries, and `portfolio_error` is
     the error in the AVERAGE claim over many decisions, which is the right
     quantity for a portfolio of buildings or a stock model.

     **FOUR PUBLISHED STATEMENTS CHANGE AND ONE IS REVERSED.**

     1. The figure's headline goes from "right to 0.8 percent on the design
        comparison" to **"right to 12.0 percent"**.
     2. **"On what using 25 percent less of a material saves, every method is
        exactly right" IS NOW FALSE for a single building.** Every method is
        10.0 to 13.4 percent out per pLCA; what is exactly right is the AVERAGE
        over many. The contrast Stage 2g drew from it -- that a quantity
        reduction is method-independent because it is a deterministic fraction
        of the material's own contribution -- survives only in the portfolio
        reading, and the paper must say which.
     3. **"On how often a specification cap binds the best method is 0.5
        percent out, so picking well is nearly the whole problem" IS
        REVERSED.** Per decision the best method is 30.6 percent out and the
        choice costs 23.1, so most of the error is there whatever you choose.
     4. The six now differ measurably on **16 of 16** claims rather than 15,
        because the quantity-reduction row now differs.

     **WHAT DOES NOT CHANGE.** No normal fit is the best method on any of the
     sixteen claims, and the normal is the worst on thirteen of them. The most
     expensive QUESTION is still `action`; the specific claim inside it moves
     from "how often a cap binds" to "a cap's chance of saving 5 percent", at
     25.1 percent. `metricset.claim_scorecard` and `per_unit_error`, with nine
     tests, one of which plants a method wrong by 0.2 on every unit with
     alternating signs.

175. **2026-09-24, Stage 2h. EVERY CROSSING IS FITTED BOTH WAYS AND PROSE PRINTS
     ONLY THE DIGITS THE DATA DETERMINE. THIS IMPLEMENTS DECISIONS 172 AND
     173**, which established that the constants are bracketed and that their
     intervals are narrower than the disagreement between two link functions.
     `[AUTHOR]` A standing rule for every fitted constant, settled at the close
     of Stage 2g.

     **ONE CORRECTION TO DECISION 173 IN PASSING.** Its text says the isotonic
     value sits outside the published interval on FOUR of the six constants.
     Its own table shows FIVE -- every one except the flip threshold at 10
     percent, where 0.0271 sits inside [0.0217, 0.0277]. Recomputed here from
     the tables: five. Nothing else in that entry changes.

     **THE PUBLISHED CROSSINGS ARE NOT EXTRAPOLATIONS** -- the flip thresholds
     are bracketed by binned observations either side over 22,500 calibration
     rows and the safe-lead ratios over 72,000 four-material comparisons -- so
     nothing needed re-deriving. What is wrong is the INTERVAL: it is a
     bootstrap on a fitted logistic's parameters, which says how well the data
     pin down that curve and nothing about whether a logistic is the right
     shape.

     **MEASURED, THE ISOTONIC FIT FALLS OUTSIDE THE LOGISTIC INTERVAL ON FIVE
     OF THE SIX PUBLISHED CONSTANTS**, not four as the stage prompt had it:

         constant           level   logistic     interval          isotonic
         flip threshold      0.01    0.00175  [0.00131, 0.00227]   0.00259  out
         flip threshold      0.05    0.01082  [0.00911, 0.01262]   0.01292  out
         flip threshold      0.10    0.02467  [0.02175, 0.02766]   0.02711  in
         safe-lead ratio     0.01    2.13267  [2.09133, 2.17095]   2.22271  out
         safe-lead ratio     0.05    1.64313  [1.62298, 1.66102]   1.61159  out
         safe-lead ratio     0.10    1.46019  [1.44678, 1.47196]   1.35217  out

     **THE RULE.** Keep the significant figures the two fits agree on, plus the
     first one they part company at; print both values where they still differ
     there. It is not a fixed count. On the current constants:

         safe lead, 1 pct     2.1 against 2.2
         safe lead, 5 pct     1.64 against 1.61
         safe lead, 10 pct    1.5 against 1.4
         flip, 1 pct          0.002 against 0.003
         flip, 5 pct          0.011 against 0.013
         flip, 10 pct         0.02 against 0.03

     The stage prompt's example for the 10 percent safe lead, 1.5 against 1.4,
     is reproduced exactly. **Its example for the 1 percent safe lead quoted
     2.1 alone; the rule gives 2.1 against 2.2**, because those two fits
     disagree at the printed digit just as the 10 percent pair does.

     **NOTHING IS THROWN AWAY AND NO CONSTANT MOVES.** The results tables and
     the supplement keep both fits and the interval at FULL precision, which is
     what a reader checking the work or carrying a constant downstream needs;
     the rule adds five columns and edits none. `flip.FLIP_THRESHOLDS`, which
     notebook 1 reads to turn a per-dataset weighting risk into a probability,
     is a computational constant rather than prose and is untouched.
     `flip.prose_digits`, `prose_crossing`, `crossing_precision`.

176. **2026-09-24, Stage 2h. THE INDUSTRY-AVERAGE CHECK IS DROPPED, because the
     extract contains none.** `[AUTHOR ASKED, GATED ON THE DATA]` The stage
     prompt gated it: "if the flag is absent or the coverage is too thin, say
     so and drop it rather than inferring which declarations are industry
     averages from their names."

     EC3 does carry a `declaration_type` field and the frozen extract does
     record it. **Every one of the 120,280 records reads `Product EPD`.** There
     is no industry-wide or industry-average declaration in the extract at all,
     so there is nothing to compare a category's uniform mean against.

     The idea stands as the right instrument for the question -- an
     industry-average declaration is in principle a production-weighted mean,
     which is exactly the quantity unknown market shares deprive a practitioner
     of (decision 100) -- and it is not testable on this data. Inferring which
     declarations are industry averages from their product names was refused.

177. **2026-09-24, Stage 2h. THERE ARE NO SIX-MODE DATASETS. The concern is
     retired rather than answered.** `[DELEGATED, 2h measured]` The stage
     inherited "5.6 percent of the corpus against an empirical 0.7 percent" and
     was told to report whether it matters before proposing a generation
     change.

     Measured at the bandwidth the study FITS, which is the density a reader is
     shown and the pLCA samples from (decision 82), the visible-mode
     distribution is:

         synthetic   1: 0.760  2: 0.212  3: 0.024  4: 0.004  5: 0.001   max 5
         empirical   1: 0.685  2: 0.262  3: 0.054                       max 3

     **Nothing on either arm has six visible modes.** The inherited figure is a
     SILVERMAN critical-bandwidth count on the Stage 2a-2 corpus, which has
     since been regenerated twice; decision 37 is where it comes from. What
     survives is a mild over-representation at four and five modes, 0.5 percent
     of the corpus against nothing in the real arm.

178. **2026-09-24, Stage 2h. ONE MARKET-SHARE RULE FOR BOTH ARMS, VALIDATED
     AGAINST THE TRUE MODE LABELS, AND THE RESIDUAL ARM GAP IS DISPERSION
     RATHER THAN THE RULE.** `[AUTHOR]` The stage's first item, carried from
     decision 141.

     **THE DEFECT, CONFIRMED.** The two arms drew market shares by different
     rules on the exact dimension the paper is built on. Median
     uniform-to-variable separation by size band, and the decay slope on
     log(n):

         empirical, stored flat Dirichlet   0.2138 0.1189 0.0678 0.0050  -0.449
         synthetic, stored mode coupled     0.1574 0.1260 0.0595 0.0528  -0.181
         synthetic values, reweighted flat  0.1330 0.0938 0.0375 0.0124  -0.367

     The third row is the counterfactual and it is the proof: reweighting the
     synthetic arm's OWN values by the empirical rule reproduces the empirical
     arm's decay, so the gap is the RULE and not the data.

     **THE RULE.** `weighting.coherent_weights`: cut the declarations into k
     groups, draw each group's share from a Dirichlet, split it inside the
     group. Groups stand in for the mixture components real data does not
     label, and the cut is a contiguous run of the SORTED values -- no fitting,
     and no failure mode at three declarations, which is why a fitted mixture
     was refused. The coherence parameter is

         s = rho * rank + (1 - rho) * uniform,   cut s into k contiguous runs

     with rho = 1 clustering share by coefficient and rho = 0 making membership
     random. `k = n` reproduces the old flat draw exactly, which a test pins.

     **THE BLOCK COUNT IS THE GENERATOR'S OWN AND A FIRST VERSION GOT IT
     WRONG.** The generator draws its component count uniformly from 1 to 5,
     INDEPENDENT of n. A first version of this grew the block count with n up
     to 12, which is a different weight model and which reintroduces the
     artifact the port exists to remove, because more groups at large n is more
     dilution at large n. `draw_blocks` is the faithful rule.

     **THE PROXY IS VALIDATED WHERE THE TRUTH IS KNOWN, which is available
     nowhere else.** The synthetic arm has both the true mode labels and the
     values, so a contiguous cut can be scored against the weights the true
     labels give, at the same block count. Over 600 datasets:

         rho    ratio of medians   median ratio   Spearman
         0.00        0.554             0.807        0.850
         0.25        0.652             0.866        0.921
         0.50        1.150             1.044        0.935
         0.75        1.494             1.178        0.877
         1.00        1.599             1.209        0.860

     **rho = 0.5 is where a contiguous cut of the sorted values reproduces what
     the true mode labels give** -- to within 4 percent on the typical dataset
     and 15 percent on the median -- and it is also where the proxy orders the
     categories most like the truth. That is a MEASURED anchor rather than a
     choice, and it is the value the port should use.

     **AND THE RESIDUAL ARM-TO-ARM GAP UNDER ONE RULE IS DISPERSION.** Dividing
     each dataset's separation by its own coefficient of variation, the two
     arms agree in every size band at every coherence level. At rho = 0.5:

         empirical   0.3964  0.2422  0.1547  0.1579
         synthetic   0.4021  0.2577  0.1668  0.1420

     So once the rule is shared, what is left is decision 96's law holding on
     both arms with different dispersion fed into it, and decision 138's
     dispersion shortfall is what feeds it. **The weight model and the corpus's
     dispersion are the same problem seen twice**, which is what the stage
     prompt anticipated in requiring they not be attempted separately.

     **rho = 0 IS EXPLICITLY REJECTED AS A NULL.** It is not the absence of an
     assumption; it is the claim that market share is uncorrelated with carbon
     intensity, and published production volumes contradict it: Marsh, Hattam
     and Allen (2025) put 63.75 percent of world steel on the higher-carbon
     Rest-of-World BOF route against 0.03 percent on Austrian EAF, and KL2's
     steel example puts 54 percent of global production in China alone.

     **WHAT IS NOT DONE HERE.** The empirical arm is NOT reweighted in the
     production path and no reported number moves. Doing so would move every
     weighted characteristic of the empirical arm, which the generator is
     calibrated against, and reopening generation is an author decision.
     `audits/weight_model.py`, `TABLE_WeightProxyValidation.csv`,
     `TABLE_WeightStatusQuo.csv`, `TABLE_WeightArmGap.csv`.

179. **2026-09-24, Stage 2h. THE CALIBRATION DOES NOT MOVE BEYOND ITS OWN
     WEIGHT-DRAW NOISE, AND THAT NOISE HAD NEVER BEEN MEASURED.**
     `[DELEGATED, 2h measured]` The stage prompt required the calibration
     consequence measured against the seed-to-seed noise. It has to be measured
     against a DIFFERENT noise, and finding that out is most of the result.

     The generator's 0.0066 seed-to-seed standard deviation is about the
     GENERATOR's seed. Every calibration cell here redraws the empirical arm's
     market shares, which is a second source of movement nobody had quantified.
     Six independent weight streams per rule, whole arm each time:

         rule                              objective        w_v_uw distance
         today: flat Dirichlet over n      0.2355 +/- 0.0148   0.1338 +/- 0.0727
         ported, rho = 0.0                 0.2372 +/- 0.0064   0.1578 +/- 0.0331
         ported, rho = 0.25                0.2453 +/- 0.0115   0.2076 +/- 0.0469
         ported, rho = 0.5                 0.2490 +/- 0.0060   0.3415 +/- 0.0930

     **THE WEIGHT-DRAW NOISE IS 0.006 TO 0.015, which is as large as or larger
     than the generator's seed noise.** So the objective's movement from
     today's rule to the ported rule at rho = 0 is 0.0017 against a noise of
     0.015 and is not a move at all; at rho = 0.5 it is 0.0135, about one
     noise standard deviation.

     **On the paper's central quantity the move IS real and is in the awkward
     direction**: the arm-to-arm distance on `w_v_uw_wasserstein` goes 0.134 to
     0.342, which is about two and a half of its own noise. That is decision
     175's finding arriving from the calibration side -- the empirical arm is
     more dispersed than the corpus, so the same rule produces a larger effect
     on it -- and it is not an argument against the rule.

     **A LATER STAGE MEASURING A CALIBRATION MOVE MUST QUOTE THIS NOISE AND NOT
     THE 0.0066.** `TABLE_WeightCalibrationNoise.csv`,
     `TABLE_WeightModelCalibration.csv`.

180. **2026-09-24, Stage 2h. A PER-DATASET WEIGHTED STATISTIC CARRIES ABOUT
     FORTY-SEVEN PERCENT RELATIVE SPREAD ACROSS WEIGHT REALIZATIONS; THE
     ARM-LEVEL VERSION OF THE SAME STATISTIC CARRIES FIVE.** `[AUTHOR]` The
     stage prompt flagged this as the item that could move a headline, and
     required both quantified and each claim assigned to one.

     Twenty-five independent weight realizations of the whole empirical arm.
     Per dataset, across realizations:

         characteristic        median   typical sd   worst range   rel sd
         w_v_uw_wasserstein    0.1076     0.0459        1.3725      0.474
         coeffvar              0.6588     0.0798        2.8136      0.120
         kurtosis              3.7683     3.0545      904.0441      0.709
         skewness              1.7230     0.4898       30.8308      0.290
         weight_outliers       0.0514     0.0371        0.4484      0.713

     **THE UNWEIGHTED TWINS MOVE EXACTLY ZERO**, which is the control that says
     the measurement is picking up the weight draw and nothing else.

     Arm level, the statistics the paper actually quotes:

         statistic                              mean       sd    rel sd
         w_v_uw_wasserstein, median            0.0981   0.0052    0.053
         w_v_uw_wasserstein, mean              0.1303   0.0056    0.043
         coeffvar, median                      0.6567   0.0215    0.033
         the law's log(n) exponent            -0.4481   0.0228    0.051
         the law's log(CV) exponent            0.8970   0.0444    0.050
         Spearman with the coefficient of variation  0.5730  0.0433  0.076
         Spearman with log(n)                 -0.5122   0.0375    0.073

     **SO THE ANSWER IS A FACTOR OF NINE.** A claim about ONE category carries
     a 47 percent relative spread and must be stated as a distribution rather
     than a number; a claim about the arm -- a median, a correlation, the
     size-and-dispersion law's exponents -- carries 3 to 8 percent and is safe
     to quote. Every weighting claim the paper makes is one or the other and
     the manuscript does not currently distinguish them.

     **AND ONE PUBLISHED FIGURE IS A CASUALTY OF THE DISTINCTION.** Decision
     96's law reaches an R2 of 0.991, and that is on the MEDIAN separation over
     1,000 draws. On a single realization the same law explains **0.824 +/-
     0.029**, and the Spearman correlation with dispersion is **0.573** against
     decision 94's 0.731 on the median. The stored characteristic
     `w_v_uw_wasserstein` IS a single realization, so a reader recomputing
     decision 96's law from the published characteristic table will not
     reproduce 0.991 and should not expect to. **The manuscript must say which
     of the two any quoted R2 or correlation came from.**
     `TABLE_WeightRealizations.csv.gz`, `TABLE_WeightRealizationStats.csv`.

181. **2026-09-24, Stage 2h. `PROFILE_DELTA_LO_FRAC` STAYS AT 0.25, AND THE
     OTHER TWO BOUNDS ARE MEASURABLY NON-BINDING RATHER THAN ASSUMED SO.**
     `[DELEGATED, 2h measured]` Swept with the tail term in force throughout,
     as Stage 2g required, reporting the largest fitted-model standard
     deviation beside W1 at every point.

         empirical      at guard  interior  max model sd  pct blowup  mean W1
             0.01        21.1      69.4         3281        20.4      0.3020
             0.05        32.0      57.8           77         14.3      0.1839
             0.10        38.1      51.0           17.9        5.4      0.1687
             0.25        47.6      41.5            3.363      0.0      0.1689
             0.50        58.5      30.6            1.869      0.0      0.1813
             1.00        69.4      19.7            1.964      0.0      0.2030

         synthetic
             0.01        11.4      72.2         5345         10.6      0.1910
             0.25        28.6      54.0            2.864       0.0      0.1004
             1.00        53.2      29.4            1.331       0.0      0.1099

     **0.25 IS THE SMALLEST GUARD AT WHICH NO FIT EXCEEDS FIVE TIMES THE
     DATA'S OWN SPREAD ON EITHER ARM**, which is the criterion decision 51
     chose it on, and with the tail term in force it is also at or beside the
     W1 minimum on both arms -- 0.1004 on the synthetic arm against 0.1027 at
     0.5 and 0.1056 at 0.1. So the choice still costs nothing on the study's
     own criterion and is still not tuned to it.

     **THE OTHER TWO BOUNDS WERE NEVER MEASURED AND BOTH ARE FINE.**
     `PROFILE_DELTA_HI_FRAC` at 10, 100, 1,000 and 10,000 gives an IDENTICAL
     largest model standard deviation and a mean W1 that stops moving above
     100, so the shipped 1,000 is an order of magnitude clear of binding.
     `PROFILE_GRID_POINTS` at 100, 200, 400, 800 and 1,600 gives identical
     numbers to four decimal places, so the shipped 400 is four times more
     than the fit needs. Neither is a knob anything turns on.

182. **2026-09-24, Stage 2h. AN UPPER TRUNCATION AT TWO TO THREE TIMES THE
     LARGEST OBSERVATION REMOVES THE RUNAWAY-TAIL FAILURE MODE AND COSTS
     ALMOST NOTHING, AND IT IS NOT ADOPTED HERE.** `[DELEGATED, 2h measured;
     ADOPTING IT IS AN AUTHOR DECISION]` Decision 152 stated the option rather
     than implementing it and handed the sweep here.

     `families.TruncatedAbove` caps any fitted model including the kernel
     estimate, and `cap_models` reads the cap off the data as a multiple of
     the largest observation, which is the only anchor available once every
     dataset is normalized to a mean of 1.0. The tail term stays in force
     throughout.

     **IT DOES WHAT IT PROMISES, AND AGAINST THE TRUTH IT IS SLIGHTLY BETTER
     THAN NOT CAPPING.** On 400 synthetic datasets, scored against the
     market-weighted true parent:

         cap      mean W1 against truth   largest model spread   in-sample W1
         1.0x           0.1957                   1.446              0.1482
         1.5x           0.1921                   1.725              0.1492
         2.0x           0.1917                   1.753              0.1502
         3.0x           0.1916                   1.900              0.1508
         5.0x           0.1917                   2.205              0.1513
        10.0x           0.1918                   2.542              0.1514
        none            0.1918                   5.404              0.1515

     **A cap at two to three times the largest observation cuts the worst
     fitted spread from 5.40 to about 1.8 and is not worse against the truth
     -- it is better in the fourth decimal.** Capping at the largest observed
     value itself is clearly harmful, 0.1957 against 0.1918, so the cap must
     be a multiple and not the maximum. On the real categories the same cap
     takes the worst spread from 3.50 to 1.99.

     **THE FAILURE IS CONCENTRATED WHERE STAGE 2g SAID IT WAS.** Uncapped, the
     largest spreads belong to the equal-weighted lognormal and the
     equal-weighted kernel estimate; the normal never exceeds 1.0 because it
     has no tail to run away with.

     **THE DECISION IS THE AUTHOR'S** because adopting it would move every
     number in the study for the sake of a failure mode two existing guards
     already keep out of the results. What this stage establishes is that the
     cost is not a cost: the choice is now between two measured options rather
     than between a measured one and an unknown.

     **MEASURED ON `corpus_2026-09-21` AND THE PRE-PORT EMPIRICAL WEIGHT RULE,
     BOTH OF WHICH WERE REPLACED LATER IN THE SAME STAGE (decisions 190, 197).**
     The ORDERINGS in this entry stand: Stage 2h established that method
     orderings are stable across the regeneration, reproducing every size-band
     winner and the practitioner threshold unchanged (decision 198). **The
     ABSOLUTE LEVELS do not**: every absolute distance in the study rose 16 to
     45 percent on the new corpus, purely because it is more dispersed. Quote
     the direction and the comparison, not the level, until the audit behind
     this entry is re-run. `audits/` holds the script; each takes 8 to 35
     minutes.

183. **2026-09-24, Stage 2h. RESOLVING THE CATEGORIES INTO PRODUCTS DID NOT
     MANUFACTURE THE HEADLINE, and the empirical arm still cannot measure a
     size crossover.** `[DELEGATED, 2h measured]` The stage prompt asked for
     the unsplit arm against every headline figure rather than against the
     aggregate, because it is the evidence that Stage 2a-3's splits did not
     produce the result.

     Three arms: the primary 147 datasets and 116,766 values; the UNSPLIT
     original EC3 categories, 136 and 119,448; and one record per
     (manufacturer, product name), 147 and 65,839. The deduplication key is
     metadata and never a value, which is the constraint decisions 43, 46 and
     60 impose on the category rules.

     **THE WIN SHARE IS ESSENTIALLY UNCHANGED AND THE ORDERING IS IDENTICAL**,
     cross-validated:

         method                      primary   unsplit   deduplicated
         Lognormal, Uniform           0.362     0.358       0.386
         KDE, Uniform                 0.260     0.236       0.244
         Lognormal, Variable          0.150     0.203       0.142
         Normal, Uniform              0.118     0.098       0.134
         Normal, Variable             0.063     0.057       0.071
         KDE, Variable                0.047     0.049       0.024

     **The LEVELS are higher on the unsplit arm and that is the expected
     direction**: median cross-validated W1 for the equal-weighted lognormal is
     0.2613 primary against 0.3006 unsplit, because an unsplit category mixes
     products. Its characteristics move the same way, median coefficient of
     variation 0.675 against 0.815.

     **AND THE SIZE CROSSOVER CANNOT BE MEASURED ON THIS ARM AT ALL**, which is
     worth stating because a reader will look for it. It reads 274 primary, 221
     unsplit and 463 deduplicated under equal weights, and 661 / 334 / 3377
     under Dirichlet-drawn shares. That is decision 136's instability -- 127
     categories cannot support this model -- and not a sensitivity to the
     population. It is also a different quantity from the corpus's 81, which is
     the conflation decision 163 warns about.

184. **2026-09-24, Stage 2h. THE JUDGMENT ARM: a pedigree model's SPREAD barely
     matters and its CENTER decides everything, and sweeping the geometric
     standard deviation alone would have missed that.** `[AUTHOR]` Decision
     124 asked for the sweep and the stage prompt required two dimensions.

     `src/judgment.py` holds the pedigree matrix as a lognormal specified by a
     center and a geometric standard deviation, plus a uniform and a triangular
     over a range derived from the SAME two inputs, so the three differ in
     SHAPE and not in information. They are not in the main comparison, for the
     reason decision 151 gives.

     **AT THE FIT LEVEL A JUDGMENT MODEL IS 2 TO 100 TIMES FURTHER FROM THE
     TRUTH** than the best data-driven fit. The realistic model, the center
     drawn as one random declaration, is 4.1 to 12 times worse.

     **AT THE DECISION LEVEL A WELL-CENTERED ONE IS COMPETITIVE.** Error in
     P(B beats A) per design pair, 300 pairs: the six data-driven methods span
     0.080 to 0.114, and a pedigree model centered on the category mean reads
     0.097 at a matched spread and **0.097 to 0.121 across a SIX-FOLD range of
     spread**. The spread axis is nearly flat. That attenuation between the two
     levels is the same one decision 166 records for data-driven methods.

     **WHAT BREAKS IT IS THE CENTER, AND THE MODE OF THE OFFSET MATTERS MORE
     THAN ITS SIZE.** A displacement applied to every material alike cancels
     EXACTLY -- every common-offset cell returns the same number to four
     decimals, because both design options' totals scale by the same factor.
     One drawn PER MATERIAL does not: 0.111 at 10 percent of the mean, 0.159 at
     25, 0.223 at 50. **Sweeping only a common offset would have reported a
     null that was an artifact of the sweep.**

     **AND THE REALISTIC MODEL IS THE BAD CASE**: one random declaration per
     material reads 0.170 to 0.271, one and a half to two and a half times the
     worst data-driven method.

     **THE SHAPE MATTERS TOO.** At a matched spread and a correct center the
     pedigree lognormal reads 0.097 where a uniform reads 0.206 and a
     triangular 0.198.

     **THE DELIVERABLE SENTENCE.** A judgment-driven model with a plausible
     spread gives the same design answer as a data-driven one provided its
     point estimate is not displaced; what a practitioner actually has -- one
     declaration per material -- displaces it independently for each material,
     and that roughly doubles the error.

     **A SOURCING GAP THE MANUSCRIPT MUST CLOSE.** The pedigree matrix's
     uncertainty-factor table is NOT in `refs/`, so the spread is swept
     RELATIVE to the data's own geometric standard deviation rather than in
     absolute pedigree units. That answers decision 124's deliverable without
     asserting a table, and it means **a specific pedigree score cannot be laid
     on this axis until the table is sourced.** Decision 49's amendment is why
     this matters: this project has already had to withdraw one figure quoted
     from memory.

     **MEASURED ON `corpus_2026-09-21` AND THE PRE-PORT EMPIRICAL WEIGHT RULE,
     BOTH OF WHICH WERE REPLACED LATER IN THE SAME STAGE (decisions 190, 197).**
     The ORDERINGS in this entry stand: Stage 2h established that method
     orderings are stable across the regeneration, reproducing every size-band
     winner and the practitioner threshold unchanged (decision 198). **The
     ABSOLUTE LEVELS do not**: every absolute distance in the study rose 16 to
     45 percent on the new corpus, purely because it is more dispersed. Quote
     the direction and the comparison, not the level, until the audit behind
     this entry is re-run. `audits/` holds the script; each takes 8 to 35
     minutes.

185. **2026-09-24, Stage 2h. EVERY GENERATOR-SHAPE PARAMETER IS ALREADY AT ITS
     BEST SWEPT VALUE. The two exceptions are both the calibration objective
     asking for the WRONG FIX, and one inherited expectation is reproduced on
     its own characteristic and does not survive the whole objective.**
     `[DELEGATED, 2h measured]` Sixteen configurations through the project's
     existing tuning harness rather than a second implementation of the
     objective, so a second copy cannot drift from the first. 440 datasets
     each; the objective's seed-to-seed standard deviation is 0.0066.

         configuration            objective   vs default   coeffvar   worst
                                              in seed sd   distance   characteristic
         current default            0.2251        --         0.380    fit_lognorm_SF
         min_q1_over_iqr 0.05       0.2301       +0.8        0.207    w_v_uw
         min_q1_over_iqr 0.2        0.2317       +1.0        0.283    fit_lognorm_SF
         min_q1_over_iqr 1.0        0.2446       +3.0        0.525    coeffvar
         min_mode_sd_frac 0.05      0.2251        0.0        0.380    fit_lognorm_SF
         min_mode_sd_frac 0.10      0.2251        0.0        0.380    fit_lognorm_SF
         min_mode_sd_frac 0.25      0.2404       +2.3        0.384    fit_lognorm_SF
         trunc_iqr_mult 2           0.2516       +4.0        0.404    entropy
         trunc_iqr_mult 5           0.2576       +4.9        0.377    fit_lognorm_SF
         trunc_iqr_mult 8           0.2809       +8.5        0.382    fit_lognorm_SF
         mode_coupling 0.0          0.2123       -1.9        0.360    entropy
         mode_coupling 0.5          0.2057       -2.9        0.351    entropy
         mode_share_alpha 1         0.2582       +5.0        0.431    w_v_uw
         mode_share_alpha 3         0.2275       +0.4        0.386    fit_lognorm_SF
         point_weight_alpha 0.3     0.2890       +9.7        0.446    w_v_uw
         point_weight_alpha 3.0     0.2468       +3.3        0.398    entropy

     **`min_q1_over_iqr`: STAGE 2a-3's EXPECTATION IS REPRODUCED AND DOES NOT
     SURVIVE.** That stage measured 0.05 and reported the
     coefficient-of-variation distance improving from 0.273 to 0.199 with the
     objective flat. Here it improves from **0.380 to 0.207**, which is the
     same finding on the same characteristic -- and the OVERALL objective is
     0.8 seed standard deviations worse, because the arm-to-arm distance on
     `w_v_uw_wasserstein` more than DOUBLES.

     **THAT IS CHECKED IN ABSOLUTE TERMS AND IS NOT A DENOMINATOR ARTIFACT**,
     which had to be established because decision 63 records a case where a
     standardized worsening was exactly that. The empirical arm is identical
     between the two configurations, so its standard deviation is fixed, and
     the two columns agree in SIGN on all ten characteristics. Undivided
     Wasserstein distance between the arms, default against the candidate:

         coeffvar             0.2685 -> 0.1463    improves by 46 pct
         w_v_uw_wasserstein   0.0285 -> 0.0583    worsens by 105 pct
         crit_bw_1            0.0787 -> 0.1053    worsens
         skewness             0.3778 -> 0.4947    worsens
         fit_norm_SF          0.0410 -> 0.0230    improves

     **It buys dispersion and pays for it with the quantity the paper is built
     on**, which is the structural trade decisions 39, 138 and 170 record on
     three other levers. Nothing was adopted then and nothing is now.

     **A NOTE ON THE WORD "WORST", because the author asked what value
     judgment it carries.** None. `tune_configuration.score` reports
     `worst_metric` as the first row of `coverage.distribution_comparison`
     sorted by standardized distance descending, so it means "the
     characteristic on which the two arms' DISTRIBUTIONS sit furthest apart",
     not "the least important characteristic". The standardization is
     necessary rather than a judgment: the characteristics are in
     incomparable units -- a distance in dataset SIZE runs to 848 while one in
     a Shapiro statistic runs to 0.04 -- so an unstandardized mean over them is
     meaningless and is dominated by `n` alone.

     **`min_mode_sd_frac` BELOW 0.15 IS NOT A PARAMETER**, which is why 0.05
     and 0.10 return the objective to the last digit and every other column
     with it. The rejection it controls fires 0 times in 300 parent draws at
     0.05, ONCE at the shipped 0.15, 10 times at 0.25 and 103 at 0.5. The
     shipped value sits exactly where the floor begins to bite.

     **`trunc_iqr_mult` SATURATES, AND DECISION 42'S LATENT BUG STAYS FIXED.**
     That decision warned the uniform-grid defect in
     `MixtureParent.truncated_moments` "would have bitten any Stage 2h sweep of
     `trunc_iqr_mult`", because raising it widens the bounds that triggered it.
     Checked over 250 parents at each of 2, 3, 5, 8 and 12: **zero degenerate
     parents at every setting**, smallest standard deviation 0.145 to 0.197.
     And the median achieved coefficient of variation stops moving above a
     multiple of about 5 -- 0.668, 0.697, 0.769, 0.769, 0.769 -- so this is not
     the lever for the dispersion shortfall either.
     `audits/truncated_moments_check.py`.

     **`mode_share_alpha`: THE AUTHOR'S PROPOSED 10 TO 1 IS 5.0 SEED STANDARD
     DEVIATIONS WORSE, WHICH REPRODUCES STAGE 2f EXACTLY AND STILL DOES NOT
     SETTLE IT.** Decision 141 recorded that same 5.0 and said the comparison
     "used the mismatched weight rules and must be redone once the arms agree".
     It is redone here under the SAME mismatched rules, because the arms still
     do not agree -- applying the ported rule is the author's decision -- so
     **the item remains genuinely blocked on that decision and not on a
     measurement.** Worth knowing meanwhile: 10 to 3 is free, at +0.4.

     **`point_weight_alpha`: THE SHIPPED 1.0 IS BEST AND CONCENTRATING IS THE
     WORST THING TRIED.** Dropping it to 0.3, which concentrates weight within
     a mode, is 9.7 seed standard deviations worse and the worst configuration
     in the sweep, with `w_v_uw_wasserstein` its worst characteristic.

     **AND THE TWO CONFIGURATIONS THAT BEAT THE DEFAULT ARE BOTH THE OBJECTIVE
     ASKING FOR THE WRONG FIX.** `mode_coupling` at 0.0 and 0.5 improve it by
     1.9 and 2.9 seed standard deviations. Every unweighted characteristic is
     untouched -- the mode-count total variation and the visible-mode shares
     are identical to the last digit -- so this is purely the weights.

     The calibration objective measures how well the two arms AGREE, and the
     cheapest way to agree is for both to make the same false assumption.
     `mode_coupling = 0` makes the synthetic arm draw weights independently of
     the values, which is exactly the empirical arm's current rule and exactly
     the claim published production volumes contradict; it would also destroy
     the market-weighted parent that every run against the truth is scored
     against. **Decision 178 ports the rule in the other direction for that
     reason, and this is the measurement showing what the objective would have
     chosen if left to itself.**

     **A LATER STAGE MUST NOT READ THOSE TWO ROWS AS A TUNING OPPORTUNITY.**
     `audits/GENERATOR_SWEEP.md` carries the commands and this warning.


186. **2026-09-24, Stage 2h. ONE OF THE SIXTEEN SCORECARD CLAIMS IS AN IDENTITY
     OF ANOTHER, AND THE OLD AVERAGED DEFINITION HID IT.** `[DELEGATED, 2h
     found while checking the corrected figure]`

     After the per-unit correction of decision 174, "what using 25 percent less
     of a material saves" and "a material's share of the building total" carry
     IDENTICAL numbers in all six cells, to six decimal places.

     **IT IS AN IDENTITY AND NOT A COINCIDENCE.** Using 25 percent less of a
     material removes exactly a quarter of that material's share of the
     building total, with no distribution entering, so

         qty_reduction_mean__error  =  0.25 x eci_perc_mean__error

     exactly. Verified over all 60,000 rows: the largest deviation is 1.1e-15
     and the correlation is 1.00000000. The true levels stand in the same
     ratio, 0.0625 against 0.2500, so the RELATIVE errors are equal.

     **THE OLD DEFINITION MADE IT INVISIBLE.** Under the averaged form both
     rows read 0.00 and 0.00, which looks like agreement between two
     independent claims rather than one claim counted twice. The correction is
     what exposed it.

     **WHAT THE PAPER SHOULD DO.** Report one of the two, not both, and say
     that a quantity reduction is a deterministic fraction of a material's own
     share so that its accuracy IS the accuracy of that share. The scorecard
     keeps both rows because the figure is grouped by the five questions a
     reader asks and a reduction strategy belongs under `action`, but the text
     must not present them as two pieces of evidence.

     **AND A CAUTION ABOUT HOW THIS WAS FOUND.** The first two checks of it
     were wrong, both because they divided by -0.25 where the saving is
     reported as a positive magnitude. One printed "CONFIRMED" unconditionally
     regardless of what it measured. A verification script that cannot fail is
     not a verification; the working check is the one that prints the residual
     and the correlation and lets the reader see them.
187. **2026-09-24, Stage 2h. THE CERTIFICATION CREDIT AS A DECISION: when a
     design sits near the bar, the six methods disagree about whether it EARNS
     THE CREDIT two thirds of the time. And the study's comparison margin
     points the WRONG WAY for this question.** `[AUTHOR]` The author's framing,
     and it is a better one than the study was using: "practitioners often
     perform LCAs for LEED points ... with probabilistic LCA, I think this
     credit would evolve to something more like 'demonstrate a 10 percent
     reduction with 75 percent confidence'."

     **WHAT THE STUDY ALREADY HAD AND NEVER ASKED.**
     `plca.reduction_statement` has computed P(an intervention delivers at
     least 5, 10 or 20 percent of the building) since Stage 2e, and the truth
     run scores the error in that probability. Nothing had asked the DECISION
     it is used for: is P at or above the required confidence, so the credit is
     earned? That inherits the error in the probability AND a cliff at the
     threshold, so it is a different and more fragile question.

     **TWO CONSTRUCTIONS WERE WRONG BEFORE ONE WAS RIGHT, and both are worth
     recording.**

     First, a credit is a WHOLE-DESIGN claim and not a one-material one.
     Capping a single material almost never moves a building by 10 percent, so
     asking the question of `cap_p_reduction_over_10` answers it where nobody
     is near the bar: the true distributions clear it in **0.24 percent** of
     cases. `audits/credit_threshold.py` is that attempt and is kept because
     its own output is the evidence for the point.

     Second, **`plca.comparison_statement` computes `P(a < g * b)`, so a margin
     ABOVE one LOOSENS the test**: `mci_1.2` is "the proposal is better, OR
     worse by less than 20 percent", which is a tolerance. A credit needs `g`
     BELOW one. Verified on the study's own run: at a true 20 percent saving
     `mci_1.2` reads **0.9993** against a discernibility of 0.9628, the looser
     condition and not the stricter one. **The docstring said "the share in
     which A beats B by a margin worth acting on", which describes `g < 1` and
     not what the code computes.** Corrected, with a test pinning the
     direction; `swap_run` now takes `margins`. **No reported number moves:
     `mci_1.05` and `mci_1.2` are what Marsh et al. (in press) report and are
     correct AS TOLERANCES; what was wrong is the sentence describing them.**

     **THE RESULT, on 600 design pairs at five true savings, 3,000 cases, with
     credit margins of 0.95, 0.90 and 0.80.** A credit of "beat the baseline by
     10 percent with 75 percent confidence":

         the truth earns it                            17.4 pct of cases
         at least two of the six methods disagree      18.2 pct
         the best method calls it wrong                 8.8 pct
         the worst method calls it wrong               12.5 pct  (Normal, Variable)

     **AND THE FRAGILITY IS THE THRESHOLD, NOT THE METHODS, which is the
     finding.** Splitting by how far the TRUE confidence sits from the line:

         distance from the line    cases   methods disagree
         0.00 to 0.05               305        65.3 pct
         0.05 to 0.10               281        46.3 pct
         0.10 to 0.25               800        20.5 pct
         beyond 0.25              1,614         3.4 pct

     A design comfortably over or under the bar is called the same way by all
     six; a design within five points of it is a coin toss. The same story by
     how much the design actually beats the baseline, at a 10 percent credit
     and 75 percent confidence: methods disagree on 0.2 percent of designs with
     no true saving, 6.5 percent at a true 10 percent saving, 31.3 at 15 and
     52.0 at 20.

     **THE NORMAL IS THE WORST METHOD ON 8 OF THE 15 CELLS**, the
     market-share lognormal on 5 and a kernel estimate on 2, which is the same
     ordering the rest of the study finds.

     **A SOURCING CONSTRAINT FOR THE MANUSCRIPT.** The tiers used are the
     study's own 5, 10 and 20 percent, which match the tiered structure
     certification schemes use. **The exact wording, tier and confidence level
     of any specific credit must be sourced before the paper cites one**, on
     the same grounds as decision 49's withdrawn figure and decision 184's
     pedigree table. `audits/credit_design.py`.

     **MEASURED ON `corpus_2026-09-21` AND THE PRE-PORT EMPIRICAL WEIGHT RULE,
     BOTH OF WHICH WERE REPLACED LATER IN THE SAME STAGE (decisions 190, 197).**
     The ORDERINGS in this entry stand: Stage 2h established that method
     orderings are stable across the regeneration, reproducing every size-band
     winner and the practitioner threshold unchanged (decision 198). **The
     ABSOLUTE LEVELS do not**: every absolute distance in the study rose 16 to
     45 percent on the new corpus, purely because it is more dispersed. Quote
     the direction and the comparison, not the level, until the audit behind
     this entry is re-run. `audits/` holds the script; each takes 8 to 35
     minutes.

188. **2026-09-24, Stage 2h. THE BANDWIDTH SENSITIVITY: the two arms disagree
     about the DENSITY criterion and agree about the study's own, so the
     shipped rule stands.** `[DELEGATED, 2h measured]` A sensitivity rather
     than an open choice; the rule was settled in Stage 2c on held-out
     likelihood.

     **THE ARMS DISAGREE ON THE CRITERION THE RULE WAS CHOSEN BY.** On the
     synthetic arm Scott beats Silverman on leave-one-out likelihood on **60.9
     percent** of datasets under equal weights and 62.7 under market-share
     weights, which is the opposite of the empirical arm where Stage 2c found
     Silverman winning. On the study's own criterion they do not disagree:
     mean W1 is **0.1517 for Silverman against 0.1649 for Scott** under equal
     weights and **0.0621 against 0.1016** under market-share weights.

     **THE CROSS-VALIDATED BANDWIDTH IS BEST ON HELD-OUT LIKELIHOOD BY A
     DISTANCE** -- mean -0.2006 against Scott's -0.4914 -- and is NOT better on
     W1, at 0.1569 against Silverman's 0.1517. That is decision 71's
     reconciliation arriving from a third direction: a density criterion and a
     CDF criterion want different bandwidths.

     **WHAT THE HEADLINE DOES UNDER SCOTT, which the manuscript was written
     against:** the kernel estimate's mean W1 rises from 0.1517 to 0.1649
     under equal weights and from 0.0621 to 0.1016 under market-share weights,
     the latter a 64 percent increase. So the manuscript's configuration
     understates the kernel estimate on the study's own criterion, which is the
     conservative direction for this paper's recommendation and must be stated
     as such rather than quietly corrected.

     **MEASURED ON `corpus_2026-09-21` AND THE PRE-PORT EMPIRICAL WEIGHT RULE,
     BOTH OF WHICH WERE REPLACED LATER IN THE SAME STAGE (decisions 190, 197).**
     The ORDERINGS in this entry stand: Stage 2h established that method
     orderings are stable across the regeneration, reproducing every size-band
     winner and the practitioner threshold unchanged (decision 198). **The
     ABSOLUTE LEVELS do not**: every absolute distance in the study rose 16 to
     45 percent on the new corpus, purely because it is more dispersed. Quote
     the direction and the comparison, not the level, until the audit behind
     this entry is re-run. `audits/` holds the script; each takes 8 to 35
     minutes.

189. **2026-09-24, Stage 2h. WEIBULL IS THE WEAKEST DATA-DRIVEN FAMILY AND
     CHANGES NOTHING, WHICH IS WHY IT IS WORTH REPORTING.** `[DELEGATED, 2h
     measured]` Added to blunt the objection that only two families were
     tested. Mean rank over the twelve (family, weighting) pairs, best
     estimator for each family:

         lognormal_3p, market shares   2.922
         kernel estimate, market       2.927
         lognormal_offset, market      4.953
         gamma, market                 5.278
         lognormal_2p, market          5.537
         weibull, market               6.235
         normal, market                6.468

     Weibull sits below gamma, which decision 70 already established is
     indistinguishable from the three-parameter lognormal out of sample. Under
     W1-optimal fitting it improves by a median of 11.85 percent and still does
     not reach gamma. **So the family list is not short for want of trying, and
     nothing in the paper's conclusions moves.**

     **MEASURED ON `corpus_2026-09-21` AND THE PRE-PORT EMPIRICAL WEIGHT RULE,
     BOTH OF WHICH WERE REPLACED LATER IN THE SAME STAGE (decisions 190, 197).**
     The ORDERINGS in this entry stand: Stage 2h established that method
     orderings are stable across the regeneration, reproducing every size-band
     winner and the practitioner threshold unchanged (decision 198). **The
     ABSOLUTE LEVELS do not**: every absolute distance in the study rose 16 to
     45 percent on the new corpus, purely because it is more dispersed. Quote
     the direction and the comparison, not the level, until the audit behind
     this entry is re-run. `audits/` holds the script; each takes 8 to 35
     minutes.

190. **2026-09-25, Stage 2h. THE PORTED WEIGHT RULE IS NOW APPLIED TO THE
     EMPIRICAL ARM IN THE PRODUCTION PATH. This SUPERSEDES decision 178's
     closing paragraph**, which built the rule, measured it, and explicitly did
     NOT apply it. `[AUTHOR]` "Why haven't you pulled the trigger on our new
     method of splitting empirical into modes and applying flat Dirichlet by
     mode? What are we waiting for? Sounds like you already figured out rho
     should be 0.5, so what do we need to discuss?"

     `empirical.WEIGHT_RHO = 0.5` and `empirical.prepare` now draws market
     shares through `weighting.coherent_weights` instead of a flat Dirichlet
     over every declaration. Both arms are on one rule for the first time in
     the project.

     **WHAT MOVED, and the control is the reason it can be trusted.** All
     twelve UNWEIGHTED characteristic columns are BIT-IDENTICAL, because the
     change touches only the weights. All twelve weighted ones move, by a
     median absolute change of 0.066 on the coefficient of variation, 0.066 on
     the uniform-to-variable Wasserstein distance, 0.135 on entropy and 0.648
     on skewness. All six W1 scores move, which is correct and not a surprise:
     every model is scored against the VARIABLE-weighted empirical CDF, so
     changing the weights changes the target.

     Arm-level, measured on the 147 real categories:

         w_v_uw_wasserstein median   0.1048 -> 0.1475
         w_v_uw_wasserstein mean     0.1348 -> 0.1898
         coeffvar median             0.6706 -> 0.6406
         coeffvar_uw median          0.678553 -> 0.678553   (the control)

     **AND IT CLOSES THE GAP IT WAS BUILT TO CLOSE.** Decay of the median
     separation on log10(n), and the medians above a thousand declarations:

         rule                         empirical   synthetic   1000+ medians
         old: flat over declarations    -0.449      -0.181     0.0050 / 0.0528
         ported, rho = 0                -0.412      -0.349     0.0064 / 0.0166
         ported, rho = 0.5              -0.161      -0.101     0.0441 / 0.0848

     The tenfold gap above a thousand declarations that decision 141 opened
     this whole item with is now a factor of 1.9, and what remains is
     dispersion rather than the rule, which decision 178 established by
     dividing each dataset's separation by its own coefficient of variation.

     **THE REGRESSION FIXTURES ARE RE-FROZEN**, three of them:
     `TABLE_EmpiricalECCMetrics.xlsx` and `TABLE_EmpiricalECCMetricsAndW1.xlsx`
     for the reason above, and `TABLE_SyntheticECCMetricsAndW1.xlsx` for an
     unrelated and benign reason -- it gained `modality_index_fitted` and
     `modality_index_fitted_uw`, the columns decision 134 settled, and no
     shared value moved by more than 1e-12. `tests/fixtures/README.md` records
     what moved and the unweighted-column control.

     **A SIDE EFFECT WORTH KEEPING.** The arm-to-arm distance on
     `fit_lognorm_SF`, which decision 37 recorded as the worst characteristic
     in the project and structural, HALVES from 0.65 to 0.33. Nothing was tuned
     to achieve that.

191. **2026-09-25, Stage 2h. A REGENERATION ON A WIDENED CONFIGURATION FAILED,
     THE FIRST TWO DIAGNOSES OF THE FAILURE WERE BOTH WRONG, AND THE CAUSE WAS
     THE TRUTH RUN'S SAMPLER RATHER THAN THE GENERATOR.** `[AUTHOR ASKED FOR
     THE REGENERATION; THE MEASUREMENT REVERSED IT]`

     The configuration was `min_q1_over_iqr = 0.02`, `trunc_iqr_mult = 5.0`,
     `cv_log10_mean = 0.429`. Its truncation bound multiplier,
     `(1 + 1/floor) ** mult`, is **345,025,251**, and the median truncation
     bound came out at 101 million on data normalized to a mean of 1.0. Every
     sample-level check PASSED and the calibration objective IMPROVED. The run
     against the true parents then reported 99.98 percent errors.

     **THE FIRST DIAGNOSIS, that the parents were tail-dominated, IS WRONG.**
     `genconfig.max_parent_mean_over_median` was added on it and does not fire
     on that configuration: its parents have a mean of 1.012, a median of
     0.773, a ratio of **1.31** against a threshold of 25, and a 1 - 1e-6
     quantile at 33. The parents were sound.

     **THE SECOND DIAGNOSIS, that quantile endpoints alone would fix the grid,
     was also wrong**, and the fix took two passes for a reason worth keeping:
     linearly-spaced points between quantile endpoints still smear 1e-4 of
     probability across eight orders of magnitude.

     **THE CAUSE.** `plca.ParentSampler` tabulated the parent's CDF on a
     LINEARLY spaced grid between the truncation bounds. With `hi` near 1e8 a
     20,001-point linear grid has a spacing of about 18,000, so the entire body
     fell between the first two grid points, the tabulated CDF became a step
     function, and inverting it returned draws spread over the whole support.
     The truth run was drawing from a lattice rather than from the parent. The
     grid now places log-spaced points into each tail between 1e-4 and 1e-12
     and spends the rest on the body, with the exact quantiles as endpoints.

     **THE CONFIGURATION AND THE CORPUS WERE BOTH REVERTED** and the shipped
     values restored. Nothing in the paper moved.

     **THE LESSON THAT BELONGS IN THE NEXT STAGE'S HANDS, because it cost a
     day.** Every check that stage ran looked at the GENERATOR or at the
     SAMPLE. Nothing looked at the object the truth run actually draws from,
     which is neither. Decision 192 is the gate that closes it.

192. **2026-09-25, Stage 2h. A CANDIDATE GENERATOR CONFIGURATION NOW HAS TO
     PASS A PARENT-LEVEL GATE BEFORE ITS CALIBRATION SCORE MEANS ANYTHING, AND
     THE REJECTED CONFIGURATION FAILS IT.** `[AUTHOR]` "Sounds good, let's make
     these edits at the parent level."

     `audits/parent_sampler_fidelity.py` compares `plca.ParentSampler`'s
     interpolated inverse CDF against the parent's own bisection at thirteen
     probabilities from 1e-6 to 1 - 1e-6, over four dataset sizes and both
     weighting schemes, then draws 10,000 values and checks the realized mean
     against an exact mean computed on a DIFFERENT node set -- the components'
     own quantiles, which is the construction decision 42 settled -- so the
     check is independent of the thing it checks.

         configuration        bound multiplier   median width   worst q error
         shipped f0.5 m3.0                  27             18          4.7e-4
         f0.2 m3.0                          64             86          5.1e-4
         f0.1 m2.0                         121             50          5.0e-4
         REJECTED f0.02 m5         345,025,251     92,000,000           0.997

     **The rejected configuration fails on 55 percent of its parents**, a
     median quantile error of 37 percent and a worst of 99.7, ALL of it at the
     1e-6 quantile where the lower bound `q1 / r**5` sits eight orders of
     magnitude below the body. So that configuration is genuinely unusable --
     not for the reason first given, and not for the reason the guard of
     decision 191 implements, but because no practical lattice can represent a
     support spanning nine orders of magnitude.

     **The bounded candidates are indistinguishable from the shipped
     configuration at the parent level**, which is what licenses reading their
     calibration scores at all.

     **THE GUARD'S DOCSTRING IS CORRECTED IN PLACE.** It claimed to be "the
     guard that would have prevented Stage 2h's failed regeneration" and cited
     parents with a mean of 6,624. Neither is true. It is kept, because the
     failure mode it names -- a parent whose mean is set by mass its own sample
     will never draw -- is real and invisible at the sample level, and it costs
     one `ppf` call per draw. It must not be cited against the widened
     configuration.

193. **2026-09-25, Stage 2h. THE STRUCTURAL TRADE BETWEEN DISPERSION AND THE
     PAPER'S HEADLINE QUANTITY IS ABOLISHED BY THE WEIGHT RULE. It was an
     artifact of scoring a mode-coupled corpus against a flat-Dirichlet real
     arm. This NARROWS decisions 39, 138, 169 and 170, none of which was wrong
     on its own evidence.** `[DELEGATED, 2h measured]`

     Those four decisions record the same finding on four different levers
     across three stages: anything widening the corpus's dispersion also widens
     its uniform-to-variable Wasserstein distance past the real arm's. 36
     configurations were searched and the sign never flipped. Decision 170's
     closing paragraph named the escape -- "breaking the trade means changing
     the WEIGHT model at the same time as the shape model" -- and that is what
     decision 190 did.

     **THE BOUNDED WIDENING CANDIDATES NOW IMPROVE BOTH AT ONCE**, three seeds
     each, against the objective's 0.0066 seed noise:

         configuration        objective   vs base   coeffvar   w_v_uw
         base (shipped)          0.2216       --      0.4085   0.3400
         f0.2 m3.0 c0.229        0.1912    -4.6 sd    0.2813   0.1951
         f0.2 m3.0 c0.329        0.1840    -5.7 sd    0.2540   0.1595
         f0.1 m2.0 c0.329        0.1738    -7.2 sd    0.2051   0.1528
         f0.1 m2.0 c0.429        0.1706    -7.7 sd    0.2040   0.1564

     The absolute and standardized columns agree in sign on every
     characteristic, so this is not the denominator artifact decision 63
     warns about.

     **THE COUNTERFACTUAL IS WHAT MAKES IT A MECHANISM RATHER THAN A
     COINCIDENCE.** `audits/widening_and_weights.py` scores ONE synthetic draw
     against the empirical arm built under four weightings, so the columns
     differ only in how the real categories were weighted. Change in the
     weighting distance from the shipped configuration, in units of its own
     0.0351 seed standard deviation:

         empirical weighting      f0.2 m3.0      f0.1 m2.0
         old: flat Dirichlet      +4.4 sd        +6.8 sd     <- the trade
         ported, rho = 0.00       -1.4 sd        -0.4 sd     <- gone
         ported, rho = 0.25       -0.5 sd        +1.8 sd     <- gone
         ported, rho = 0.50       -5.1 sd        -5.3 sd     <- reversed

     **THE SIGN FLIP IS THE CHANGE OF RULE AND NOT THE VALUE OF rho.** At
     rho = 0, which is the ported rule's own null, the trade is already
     indistinguishable from zero. What rho buys on top is the objective: the
     candidates improve it by 1.8, 4.3 and 6.5 seed standard deviations at
     rho = 0, 0.25 and 0.5. The dispersion column is essentially unchanged
     across all four arms, as it must be, because the corpus's own spread does
     not depend on how the real categories are weighted.

     **WHY IT HAPPENS, in one sentence.** Porting the rule raised the real
     arm's median weighting effect from 0.105 to 0.148 while the shipped corpus
     sits at 0.092, so the corpus now UNDERSTATES that quantity by a third and
     widening moves it toward the arm instead of past it.

     **WHAT IS NOT DECIDED HERE: whether to regenerate.** Generation has been
     closed since decision 48 and every number in the paper moves when it
     reopens. What this establishes is that the reason for keeping it closed on
     the dispersion question -- that widening costs the headline quantity -- no
     longer holds, and that `crit_bw_1` is the one characteristic that worsens,
     from 0.158 to 0.259 standardized, on a characteristic carrying weight 3 in
     the objective. **A stage acting on this must judge that cost and must
     re-run the parent-level gate of decision 192 on whatever it adopts.**

194. **2026-09-25, Stage 2h. THE PEDIGREE MATRIX IS SOURCED, AND IT CANNOT
     REACH THE SPREAD OF REAL ECC DATA. This NARROWS decision 184's sweep**,
     most of which turns out to be unreachable. `[AUTHOR]` "you should've just
     told me to go fetch it! I just dropped two papers in there."

     From Muller, Lesage, Ciroth, Mutel, Weidema and Samson (2016), Int J Life
     Cycle Assess 21:1185-1196, Table 3's prior column, which is what ecoinvent
     uses, with the basic uncertainty factor of 1.05 its Table 4 gives for
     semi-finished products and materials under every sector it reports.
     `audits/pedigree_range.py` enumerates all 3,125 score combinations.

     **THE ARITHMETIC, because every factor in that table is a contributor to
     the SQUARE of the geometric standard deviation:**

         sigma_95 = sqrt(sum over indicators of [ln(UF_i)] ** 2, plus basic)
         GSD      = exp(sigma_95 / 2)

     Quoting the combined factor AS a geometric standard deviation would double
     the spread.

         best  (1,1,1,1,1)            GSD 1.0247
         median combination           GSD 1.2416
         worst (5,5,5,5,5)            GSD 1.5873
         a median real ECC category   GSD 1.8712

     **SO A PEDIGREE MODEL IS SYSTEMATICALLY NARROWER THAN THE DATA IT STANDS
     FOR, and 61.9 percent of real categories are wider than its worst score
     can reach.** End to end the matrix spans a factor of 1.55; the real arm
     spans 1.01 to 50.9. That is a property of what the matrix is FOR --
     uncertainty about one datum for one process, not the spread of products
     within a material category -- rather than a defect in it, and **the
     manuscript should say so rather than present the two as rival estimates of
     one quantity.**

     **OF THE SIX SPREAD RATIOS DECISION 184 SWEPT, ONLY 0.5 IS REACHABLE ON
     THE MEDIAN CATEGORY.** Taken the way that sweep takes it, on the EXCESS
     over 1, since `gsd = 1 + (gsd_data - 1) * ratio` and a GSD of 1 is no
     spread at all, the reachable band is 0.028 to 0.674. Ratio 0.5 is
     reachable on 62.6 percent of real categories, 1.0 on 38.1 and 3.0 on 5.4.
     The wide end of that sweep is a sensitivity and must not be labeled a
     pedigree model.

     **THIS STRENGTHENS DECISION 184 RATHER THAN UNDERMINING IT.** That entry
     found the spread axis nearly flat -- design-comparison error 0.097 to
     0.121 across a six-fold range -- and the reachable band is narrower still,
     so its conclusion that the CENTER decides everything holds with more room
     to spare.

     **TWO READING ERRORS ARE RECORDED SO THEY ARE NOT REPEATED.** A note taken
     from that paper read "GSD 1.279 basic rising to 1.690 at scores 5,5,5,5,5";
     those are the posterior factors for ONE indicator, the further
     technological correlation, at scores 2 and 3 for the manufacturing sector,
     and are neither GSDs nor a range. And the first version of the audit
     reported a STRAIGHT ratio of model GSD to data GSD, giving 0.55 to 0.85
     and wrongly naming 0.75 as reachable, against a sweep that scales the
     excess. `judgment.PEDIGREE_GSD` carries the three computed values so
     nothing downstream quotes them from memory.

195. **2026-09-25, Stage 2h. THE BANDWIDTH THROUGH THE pLCA: Scott is worst on
     every output, the guard costs almost nothing, and the parametric controls
     do not move at all.** `[AUTHOR]` "Ultimately, pLCA results are most
     important, so should we measure those? Silverman vs scott vs guarded
     silverman?" Decision 188 answered on the FIT; this answers on the answer.

     2,000 pLCA groups against the true parents, mean relative error over five
     outputs, in percent:

         bandwidth rule       KDE, Uniform   KDE, Dirichlet shares
         scott                    25.879            25.418
         silverman                25.283            24.965
         silverman_guarded        25.473            24.819

     **THE CONTROL IS THAT THE FOUR PARAMETRIC METHODS ARE BIT-IDENTICAL ACROSS
     ALL THREE RULES**, which they must be and which says the measurement is
     picking up the bandwidth and nothing else.

     **Scott is worst on every one of the five outputs under both weightings**,
     which is decision 71's finding reaching the decision level. The guard
     costs 0.19 of a percentage point under equal weights and BUYS 0.15 under
     Dirichlet shares, so at the decision level it is free in a way it is not
     at the fit level. **The shipped rule stands and nothing changes.**

     Worth stating in the manuscript: the configuration it was written against
     used Scott, so the paper's own numbers understate the kernel estimate on
     every downstream output, which is the conservative direction for its
     recommendation.

196. **2026-09-25, Stage 2h. TWO PROMISED CHECKS, BOTH CLEAN: the characteristic
     that WORSENS under widening is worth a quarter of the one that improves,
     and the upper truncation does not pull the two arms apart -- though it
     does pull the two FAMILIES apart in sample.** `[DELEGATED, 2h measured]`

     **THE FIRST.** Decision 193 leaves one characteristic moving the wrong way
     under the bounded widening candidates: Silverman's critical bandwidth,
     0.158 to 0.259 standardized, on a characteristic carrying weight 3 in the
     calibration objective. "A characteristic moved" is not a reason to act
     until it is shown to move a conclusion, which is decision 82's test and
     which `audits/dispersion_matters.py` now applies to any characteristic
     rather than only to dispersion.

     Reweight the corpus so its distribution of one characteristic matches the
     real arm's, and see whether the method comparison moves:

         characteristic   weighting    KDE win share       shift
         coeffvar         equal        0.6545 -> 0.6287   -0.0258
         coeffvar         Dirichlet    0.7033 -> 0.6608   -0.0425
         crit_bw_1        equal        0.6545 -> 0.6485   -0.0060
         crit_bw_1        Dirichlet    0.7033 -> 0.6881   -0.0152
         fit_lognorm_SF   equal        0.6545 -> 0.6531   -0.0014
         fit_lognorm_SF   Dirichlet    0.7033 -> 0.7048   +0.0015

     **The characteristic that worsens moves the comparison about a QUARTER as
     much as the one that improves**, in the same direction, so the widening
     trade is favorable on this evidence rather than merely favorable on the
     objective. **And `fit_lognorm_SF` moves it by essentially nothing**, which
     is worth recording because decision 37 called that characteristic the
     worst in the project and structural, and four stages have worried about
     it. No bin is empty on any of the three, so the reweighting is fully
     supported rather than extrapolating.

     **THE SECOND.** Decision 182 measured an upper truncation of each fitted
     model and left a promise to check it does not affect the two arms
     differently, which would make them less comparable on the dimension the
     study compares them on. It does not. In-sample W1, mean over all six
     methods, relative change from the uncapped fit:

         cap        empirical   synthetic   differential
         1.0x         -2.54 pct   -2.19 pct     -0.35 pp
         2.0x         -0.99       -0.87         -0.12
         3.0x         -0.50       -0.42         -0.08
         5.0x         -0.18       -0.14         -0.03

     **BUT IT IS NOT NEUTRAL BETWEEN FAMILIES, and that is the finding worth
     keeping.** At a cap of 2x the largest observation the whole effect is on
     the lognormal -- -3.5 and -4.1 percent on the empirical arm, -2.3 and -3.8
     on the synthetic -- while the kernel estimate and the normal move by less
     than 0.03 percent, because neither puts any mass beyond the data to cap.
     That is the same asymmetry decision 85 records for the tail term, and for
     the same reason.

     **AGAINST THE TRUTH THE ASYMMETRY LARGELY VANISHES, which is what settles
     it.** Scored against the market-weighted true parent rather than in
     sample, a cap of 2x moves `Lognormal, Uniform` by +0.57 percent and
     `Lognormal, Variable` by -0.85 -- opposite signs, both under one percent
     -- and the kernel estimate and the normal by 0.00. The paired
     kernel-minus-lognormal difference the paper reports goes from -0.00295 to
     -0.00392 under equal weights and from -0.00886 to -0.00738 under Dirichlet
     shares, which is the fourth decimal.

     **So the in-sample gain the cap buys the lognormal is a gain on the
     CRITERION and not on the ANSWER**, and adopting the cap would move no
     conclusion in either direction. That strengthens decision 182's finding
     that the cost is not a cost, and it adds the reason a reviewer would ask
     for: the cap is not quietly closing the gap between the two families.

197. **2026-09-25, Stage 2h. THE CORPUS IS REGENERATED AS `corpus_2026-09-25`,
     WIDENING THE ONE CHARACTERISTIC IT NEVER MATCHED. Generation was reopened
     by author decision, and decisions 47, 48, 55, 138, 169 and 170 are
     superseded on the question of whether this is possible.** `[AUTHOR]`
     "We agreed that the synthetic corpus isn't doing its job correctly,
     right? So why wouldn't we remake the synthetic corpus?"

     `min_q1_over_iqr` 0.5 to **0.2** and `cv_log10_mean` 0.129 to **0.329**;
     `trunc_iqr_mult` stays at 3.0. The truncation bound multiplier goes from
     27 to **216**, which is eight times wider and three thousand times
     narrower than the 345 million of the configuration this stage rejected.

     **WHY THIS BECAME POSSIBLE, and it is not that anyone tried harder.**
     Decisions 39, 138, 169 and 170 record the same trade on four levers across
     three stages: widening the dispersion always inflated the
     uniform-to-variable Wasserstein distance past the real arm's. That was an
     artifact of the two arms drawing market shares by DIFFERENT rules, and it
     disappears once they share one (decisions 190, 193). Those decisions were
     right on their own evidence; their evidence had a shared cause nobody had
     isolated.

     **THREE GATES, ALL ON THE REAL 10,000-DATASET CORPUS rather than a
     draft.**

         parent-level fidelity   5.1e-4 worst quantile error, every parent
         end-to-end smoke        mean W1 0.2314, worst fitted spread 2.38,
                                 mean draw from the true parent 1.0000
         calibration at scale    objective 0.1862, against 0.1840 predicted
                                 from 1,000-dataset drafts

     The draft prediction landing within 0.002 of the production value is what
     says the 1,000-dataset protocol of decision 41 is trustworthy for this
     kind of decision.

     **WHAT THE CORPUS NOW LOOKS LIKE, worst characteristic first, against 147
     real categories:**

         characteristic      before   after    real median / synthetic
         fit_lognorm_SF       0.233   0.341    0.9353 / 0.9290
         entropy              0.373   0.313    2.9010 / 3.3198
         crit_bw_1            0.158   0.266    0.7132 / 0.7100
         coeffvar             0.409   0.247    0.6406 / 0.6167
         n                    0.236   0.237    47 / 99
         fit_norm_SF          0.358   0.227    0.8312 / 0.8550
         w_v_uw_wasserstein   0.340   0.151    0.1475 / 0.1098
         skewness             0.219   0.137    1.7773 / 1.5870

         objective            0.2216  0.1862   -5.4 seed standard deviations

     **The two characteristics the change was for both improve**, and the two
     that worsen were both measured as low-stakes before the decision was taken
     (decision 196): matching the corpus to the real arm on `fit_lognorm_SF`
     moves the method comparison by 0.001 to 0.002 and on `crit_bw_1` by 0.006
     to 0.015, against 0.026 to 0.043 for dispersion.

     **`fit_lognorm_SF` IS NOW THE FURTHEST-APART CHARACTERISTIC and the
     manuscript's limitation paragraph must be rewritten around it** rather
     than around dispersion, which is what it currently names.

     **NUMBERS THAT MOVED: every number in the paper.** The regression fixture
     for the synthetic arm is re-frozen and `tests/fixtures/README.md` records
     every column with its before and after. The two EMPIRICAL fixtures did NOT
     move, which is the control. 588 of 591 tests passed before the re-freeze
     and all three failures were that fixture and the two tests that read it.

     **ONE READING TRAP, and it is the one a reviewer will hit.** Every
     absolute distance rises: the six W1 columns by 16 to 45 percent and the
     error in a material's estimated contribution by 25 to 35. That is SCALE,
     not degradation -- a more dispersed dataset has a wider parent and a
     larger absolute distance to it. Divided by each dataset's own spread the
     distance to the truth is **0.965** times the shipped corpus's, i.e.
     slightly better. `truncated_mass` rising from 0.096 to 0.162 through WIDER
     bounds is the same effect: more dispersed parents have more tail to lose.

     **AND THE AUTHOR CHECKED THE DATASETS BY EYE, which is not a formality in
     this project.** Having run notebook 1 to look at the example panels:
     "the sample datasets looked great." That matters because decision 38
     records three separate occasions where the calibration statistics improved
     while the generated shapes were visibly wrong, and the author caught it by
     eye each time before any objective did; decision 169 built
     `audits/corpus_examples.py` for exactly this check. A regenerated corpus
     that satisfies the objective AND looks right is a stronger acceptance than
     either alone, and this is the first regeneration in the project to have
     both on record.

198. **2026-09-25, Stage 2h. THE REGENERATION CHANGES NO RECOMMENDATION THIS
     PAPER MAKES, AND THAT IS THE RESULT.** `[DELEGATED, 2h measured]` The
     author's question on being told the corpus had been rebuilt: "Did any of
     our high level recommendations change? When is lognormal best and when is
     KDE best?"

     Every headline was recomputed on the new corpus rather than assumed.

     **THE PRACTITIONER THRESHOLD IS UNCHANGED AT 81 DECLARATIONS**, with the
     indistinguishable band 68 to 106 against decision 142's 68 to 97. The
     penalty curve is the same shape: 6.01 points of extra error at a threshold
     of 24, 0.10 at 81, 1.27 at 138, 5.96 at 304.

     **WHICH METHOD IS CLOSEST TO THE TRUTH, by dataset size, share of
     datasets. Decision 139's figures in brackets:**

         n           KDE eq      KDE Dir     Logn eq     Logn Dir    Normal
         3-9        34.7 [33.8] 22.4 [22.5] 22.6 [20.9] 10.6 [ 9.9]  9.8
         10-99      21.0 [22.5] 21.1 [20.0] 27.2 [25.2] 20.5 [17.9] 10.1
         100-999    24.9 [28.8] 38.0 [42.8] 12.2 [ 8.8] 22.2 [16.8]  2.6
         1000+      21.8 [23.5] 67.0 [69.6]  1.5 [ 0.6]  9.6 [ 6.1]  0.1

     **EVERY ORDERING IS IDENTICAL and no figure moves by more than five
     points.** The kernel estimate under equal weights leads at three to nine
     declarations, the three-parameter lognormal under equal weights leads at
     ten to ninety-nine, and the kernel estimate under Dirichlet shares leads
     everywhere above a hundred, reaching 67 percent at the top. **The normal
     is never best in any band under either weighting.**

     **AND THE TRUTH RUN'S HEADLINE SHARPENS.** Against the market-weighted
     true parent, error in a material's chance of leading: the four non-normal
     methods span **0.0811 to 0.0841**, a 3.7 percent spread against 6.8 on the
     old corpus, so they are MORE indistinguishable than decision 109 reports;
     and the normal's penalty over the best grows from 44.2 to **50.6
     percent**. `KDE, Variable` still names the true largest contributor most
     often, 0.5208 against 0.5250.

     **WHY THIS IS WORTH A DECISION ENTRY RATHER THAN A SHRUG.** These
     recommendations were calibrated on a corpus whose dispersion was known to
     be short by a factor of nine and whose generator settings have now
     changed. Reproducing them on differently-built data is the robustness
     check the paper could not previously offer, and it is a stronger claim
     than the one the manuscript currently makes on a single corpus. **The
     manuscript should say the rule was reproduced on two independently
     generated corpora.**


199. **2026-09-25, Stage 2h. THREE AUTHOR DECISIONS AT THE CLOSE OF THE STAGE.**
     `[AUTHOR]`

     **THE MANUSCRIPT DESCRIBES ONE CORPUS, NOT TWO.** "I thought we'd just
     focus on this one because it's more right. I don't think generating two
     corpora of data should be a major part of our methodology." Correct, and
     decision 198's closing recommendation is WITHDRAWN: it suggested the paper
     say the practitioner rule was reproduced on two independently generated
     corpora. It should not. The superseded corpus is the less representative
     one, and describing both invites a reviewer to ask why the worse one is
     shown. **The reproduction stays as internal verification** -- it is why
     the rule can be trusted, it is recorded in decision 198 and discrepancy
     entry 173, and a later session should not re-run it. Everything else in
     198 stands.

     **THE UPPER TRUNCATION IS NOT ADOPTED and is stated as a remedy a reader
     can apply.** "We just need to note in the manuscript that truncation is an
     option that's very easy to apply if you're dealing with extreme values."
     This closes the open item decisions 152 and 182 left. Adopting it would
     move every number a second time for a failure two guards already keep out
     of the results. `families.TruncatedAbove` and `cap_models` stay in the
     code, tested and unused, so a later stage can adopt it without rebuilding
     it. The paper gets one or two sentences in the DISCUSSION, with the
     measured figures behind them: a cap at two to three times the largest
     observation cuts the worst runaway fit from 5.4 times the data's spread to
     1.8 and is not a cost against the truth. Discrepancy entry 174.

     **THE WEIGHTING VOCABULARY IS "MARKET WEIGHTS" AGAINST "UNIFORM
     WEIGHTS".** "Market weight vs uniform weight seems right to me. Let's just
     make sure we're consistent everywhere in the language we use." This is the
     fourth vocabulary and it is the last; the churn is recorded in
     `fitting.WT_DISPLAY` so it is not repeated. It SUPERSEDES decision 160's
     "Dirichlet shares" and the "sampled market shares" that replaced it, and
     the author caught a genuine inconsistency: the code said "sampled market
     shares" while its own docstring still said "Dirichlet shares".

     **THE OBJECTION DECISION 160 RAISED IS STILL VALID AND MOVES INTO THE
     TEXT.** "Market weights" alone can be read as real production volumes,
     which this study does not have, and that reading is what made a result
     where uniform weighting wins look like a modeling error. Two things
     prevent it, neither of them a label: the methods section says at first use
     that market weights are DRAWN from a Dirichlet because production volumes
     are not published, and the oracle scheme is "known market shares", so the
     contrast between a drawn weight and a known one is visible wherever both
     appear. **A label cannot carry a caveat; a sentence can.** The stored
     `method` values are untouched -- they join every table to every fixture.
     Discrepancy entry 175.

200. **2026-09-25, Stage 2h. THE GENERATOR SWEEP RE-RUN UNDER THE SETTLED
     WEIGHT RULE: nothing beats the shipped configuration, and the two
     configurations that used to beat it are now the two worst. This
     RE-ESTABLISHES decision 185 rather than confirming it, because the
     configuration it was measured against has changed.** `[DELEGATED, 2h
     measured, at the manuscript session's request]`

     Decision 185 swept sixteen configurations on 2026-09-24, BEFORE the weight
     rule was ported and on the corpus since replaced, so every "currently" in
     it named a value that has moved. Fourteen configurations re-run at two
     seeds against the new shipped configuration and the reweighted empirical
     arm, judged against the **weight-draw noise of 0.006 to 0.015** (decision
     179) rather than the generator's 0.0066:

         min_q1_over_iqr=0.05    0.1793   -0.0105   inside the noise
         min_q1_over_iqr=0.1     0.1822   -0.0076   inside the noise
         DEFAULT (shipped)       0.1897    0.0000
         min_mode_sd_frac=0.05   0.1897    0.0000   identical, as before
         trunc_iqr_mult=2.0      0.1907   +0.0009
         mode_share_alpha=3.0    0.1974   +0.0076
         trunc_iqr_mult=5.0      0.2031   +0.0133
         point_weight_alpha=3.0  0.2036   +0.0139
         point_weight_alpha=0.3  0.2039   +0.0141
         mode_share_alpha=1.0    0.2046   +0.0149   at the edge
         min_mode_sd_frac=0.25   0.2066   +0.0169   worse
         min_q1_over_iqr=0.5     0.2086   +0.0188   worse, and it is the
                                                    configuration that shipped
                                                    until this stage
         mode_coupling=0.5       0.2175   +0.0277   worse
         mode_coupling=0.0       0.2277   +0.0379   worse

     **NOTHING BEATS THE DEFAULT BEYOND THE NOISE**, so decision 185's
     conclusion holds for the new configuration.

     **AND THE ARTIFACT DECISION 185 WARNED ABOUT IS GONE, WHICH IS THE PART
     WORTH KEEPING.** That decision recorded `mode_coupling` at 0.0 and 0.5
     beating the default by 1.9 and 2.9 seed standard deviations and warned, in
     terms, that this was the objective asking for the WRONG FIX -- the
     cheapest way for the two arms to agree was for the synthetic arm to adopt
     the empirical arm's false assumption that market share is uncorrelated
     with carbon intensity. **Under one weight rule those same two
     configurations are the two WORST of the fourteen.** The warning was
     correct and the mechanism behind it has been removed.

     **AN INDEPENDENT CHECK ON THE REGENERATION.** `min_q1_over_iqr = 0.5`, the
     value that shipped until this stage, is now measurably worse than the 0.2
     that replaced it, at +0.0188 against a noise of at most 0.015.

     **AND THE AUTHOR'S `mode_share_alpha` PROPOSAL IS NOW NEUTRAL, where
     decisions 141 and 185 both recorded it as 5.0 seed standard deviations
     worse.** Moving it from 10 to 1 measures +0.0149 against a weight-draw
     noise of 0.006 to 0.015, so it sits exactly at the edge and is not
     distinguishable from the default. Both earlier measurements were made
     under the mismatched weight rules and against the wrong noise floor.
     **It costs nothing measurable and buys nothing measurable on the
     objective**, which makes it a free choice on other grounds -- and decision
     169 found it the single most effective lever on the multimodal SHARE,
     reaching 29.7 percent against a real 31.5. Since the regeneration left the
     CONDITIONAL modality worse -- among dispersed datasets the corpus is now
     multimodal 14.1 percent of the time against the real arm's 21.2, where the
     superseded corpus matched at 22.1 -- this is worth re-opening in Stage 2j
     rather than treating as settled.

201. **2026-09-25, Stage 2h. TWO DEFECTS FOUND BY THE MANUSCRIPT SESSION AFTER
     THE STAGE CLOSED, both of them this stage's own recorded habits applied to
     its own output.** `[MANUSCRIPT SESSION FOUND, 2h FIXED]`

     **A CONFIGURATION DUMP CRASHED AND WAS PASTED WITHOUT BEING READ.** It
     raised `KeyError: 'count'` on the strata line -- the field is
     `n_datasets` -- printed the generator block and died before reaching the
     weight rule, fitting, families, customstats or `flip.FLIP_THRESHOLDS`,
     which was most of what had been asked for and included the one item the
     reader said they most needed. **This is habit 34 inverted.** That habit
     says a verification script that cannot fail is not a verification; this
     was a script that DID fail whose failure nobody read. The replacement
     ends with an explicit completion sentinel so a truncated paste is visible
     as one, and the dump is checked for a traceback before it is used.

     **A FIGURE CAPTION WAS PATCHED WHERE IT SHOULD HAVE BEEN REBUILT.** The
     scorecard figure was correctly redrawn from current tables, and its
     caption was corrected only on the five rows whose DEFINITION had changed
     -- leaving at least nine other figures from the superseded corpus in a
     paragraph that read as internally consistent. It said 12.0 where the
     table gives 12.59, 32.0 against 32.42, 10.0 against 12.22, 43.6 to 45.5
     against 47.60 to 49.76, and 8.0 and 22.1 against 9.43 and 24.19, and it
     still used the retired word "sampled".

     **THE LESSON, AND IT IS THE ONE THIS STAGE SPENT ITSELF ON.** A caption
     mixing two corpora while reading as one paragraph is the same failure as a
     scorecard mixing two statistics on one color scale (decision 157) and a
     figure mixing two error definitions (decision 174). **When the data under
     a piece of prose is replaced, REBUILD the prose from the table rather than
     repairing the numbers that are known to have moved**, because the ones not
     known to have moved are exactly the ones that will be missed. Every
     caption in the repository should be assumed to have the same defect until
     checked; Stage 3 owns that sweep and the manuscript session has written it
     up as its worked example.


202. **2026-09-25. THE TWO-SURFACE WORKFLOW IS RETIRED. Claude Code owns the
     prompt file, and the adversarial review moves inside it.** `[AUTHOR]`
     "This workflow isn't going well. The other Claude agent isn't bringing
     much value. I think we can do everything we need to do within Claude Code
     and this repository."

     **WHAT THE CHAT WINDOW WAS ACTUALLY FOR, which is worth stating because
     the replacement has to preserve it.** Not prompt editing. It had not run
     the analysis. In its last three reviews it caught that six audit results
     were computed on a corpus replaced later in the same stage; that a
     generator sweep's conclusion was measured against a contaminated
     objective; that a figure caption carried nine numbers from a superseded
     corpus while the figure itself was current; and that a configuration dump
     had crashed and been pasted with its traceback. **None of those needed
     repository access and all of them needed a reader who did not already
     believe the session's account of its own work.**

     **SO THE SEPARATION SURVIVES AND MOVES INSIDE CLAUDE CODE.** At the close
     of every stage, a FRESH window reads the stage report and attacks it before
     the next stage starts. Its standing questions are in
     `reports/START_HERE.md`. It is not optional, and it is most valuable on the
     stages that feel cleanest.

     **FILE REORGANIZATION.** `reports/START_HERE.md` is the single entry point
     and names everything else; `reports/STAGE_PROMPTS.md` holds the stage
     instructions with a line index. The invocation is one line: "Read
     reports/START_HERE.md and follow it. I am starting Stage <id>."

     **THE HANDOFF SPECIFICATION IS REPLACED BY A STAGE REPORT SPECIFICATION**,
     above. The reader now HAS the repository, so "see this file" becomes
     legitimate and a decision number can be a citation. Two requirements are
     added in exchange: **every headline claim carries the command that
     reproduces it**, and **every result states which corpus and which weight
     rule it ran on**. Both come directly from Stage 2h failures. What does not
     change is numbers as text, figures embedded, and a plain-language "so what"
     on every claim, because the author reads it without opening tables.

     **AND EVERY TABLE A STAGE WRITES NOW STAMPS ITS PROVENANCE**, the corpus
     label and the weight rule, so staleness is visible in the artifact rather
     than reconstructible from file timestamps.

203. **2026-09-25. THE CORPUS KEEPS `mode_share_alpha = 10` AND THE MODALITY
     SHORTFALL IS A STATED LIMITATION. The lever works and the price is the
     paper's headline quantity, for the fourth time.** `[AUTHOR]` "Let's keep
     bounded_mid and state the limitation."

     **WHAT THE LEVER BUYS, measured cleanly on the shipped configuration at
     1,000 datasets, against 130 real categories on the weight-invariant
     unweighted columns:**

         configuration    multimodal  dispersed  both   multi|disp  weighting
         REAL ARM            0.246      0.254   0.054     0.212        --
         SHIPPED             0.2000     0.0861  0.0127    0.1471     0.1722
         + alpha = 1         0.2418     0.0785  0.0127    0.1613     0.2751

     `mode_share_alpha = 1` takes the multimodal share from 0.200 to **0.242
     against a real 0.246**, which is essentially exact, and lifts the
     conditional from 0.147 to 0.161. **It costs the arm-to-arm distance on
     `w_v_uw_wasserstein` 0.172 to 0.275, a 60 percent degradation on the
     characteristic the paper is built on**, plus dispersion 0.260 to 0.291 and
     the objective 0.228 to 0.250.

     **THE DECISION: keep 10.** "How much does market weighting change a
     material's distribution" is what the paper is for, and a corpus that
     overstates it by 60 percent undermines the central claim in a way a
     modality shortfall in about one real category in twenty does not.

     **THE LIMITATION TO STATE, and it is now precisely founded rather than
     "we tried things".** The corpus spans the modality of real categories and
     the dispersion of real categories and under-represents their
     INTERSECTION: multimodal-and-dispersed is 1.3 percent of the corpus
     against 5.4 percent of the real arm, and conditional on being dispersed a
     real category is multimodal 21.2 percent of the time against the corpus's
     14.7. **What the paper cannot speak to is the roughly one-in-twenty real
     categories that are both.**

     **AND THE MECHANISM IS KNOWN, which is what makes it a limitation rather
     than an unknown.** The conditional is HIGHEST on the narrow superseded
     configuration (0.240) and falls in every widening candidate, because
     widening the components blends the humps together. Three earlier
     candidates combined `alpha = 1` with MORE separation and all three made
     the conditional worse still (0.139, 0.111, 0.136) -- separation is what
     destroys it, not the hump-share concentration. So the conditional and the
     dispersion marginal are in direct tension in this generator, and the
     shipped configuration trades one for the other deliberately: it bought a
     dispersion marginal that was short by a factor of nine and paid part of it
     in a conditional that was closer.

     **FOURTH APPEARANCE OF ONE TRADE.** Decision 39 in Stage 2a-2, decision
     138 in Stage 2f, decision 170 in the Stage 2g review, and here. Decision
     193 showed the dispersion-versus-weighting form of it was an artifact of
     the two arms weighting differently and removed it. **The modality-versus-
     weighting form is NOT that artifact and survives the repair.**

204. **2026-09-25, Stage 2j. LETTING THE METHOD VARY BY MATERIAL IS NOT A NULL:
     the study's own one-number rule is the closest of the seven policies on
     ALL SIXTEEN claims a probabilistic LCA makes, by a median 11.3 percent of
     the best fixed policy's own error.** `[DELEGATED, 2j measured]` The stage
     was written to expect a null and to report one as a result. It did not get
     one.

     **THE RULE, and nothing about it was re-derived.** A kernel estimate with
     market weights at or above 81 declarations, a three-parameter lognormal
     with uniform weights below. 81 is decision 142's practitioner threshold,
     reproduced unchanged on the regenerated corpus by decision 198; both the
     family and the weighting switch at the same line because decision 161
     found both orderings invert at about 100 declarations. Of the 10,000
     synthetic datasets it assigns the kernel estimate to 5,220 and the
     lognormal to 4,780.

     **NOTHING IS REFITTED.** `mixedpolicy.add_mixed` puts a NEW KEY on each
     dataset's existing fitted-model dictionary pointing at whichever of the
     six models the rule selects, so the object sampled under the mixed policy
     IS the object that fixed policy samples. **That is checked rather than
     asserted**: over the 119 pLCA groups whose four materials all sit below
     the threshold the mixed rows differ from the lognormal-with-uniform-
     weights rows by 0.00e+00 on every output, and over the 172 entirely above
     it from the kernel with market weights by 0.00e+00.

     **THE GAIN, against the best FIXED policy on each claim -- which is the
     comparator a reader would otherwise use and therefore the hard test --
     with a paired cluster bootstrap over pLCA groups:**

         a cap: how often it binds                16.76   [14.49, 19.00]
         a cap: its chance of saving 5 pct        15.41   [13.21, 17.65]
         the chance of meeting a budget           15.20   [11.74, 18.63]
         the probability B beats A                14.75   [10.85, 18.30]
         a material: its chance of being largest  14.27   [13.09, 15.49]
         a material: its share of the total       12.43   [ 9.73, 15.22]
         using 25 pct less: its mean saving       12.43   [ 9.88, 15.18]
         the total: its 90th percentile           11.39   [ 8.06, 14.87]
         a cap: its mean saving                   11.29   [ 9.49, 13.22]
         the total: its mean                       9.99   [ 6.60, 13.55]
         the total: its standard deviation         7.86   [ 5.82,  9.87]
         a material: its 95th percentile           7.72   [ 5.73,  9.72]
         a material: its mean contribution         5.75   [ 3.38,  8.10]
         a material: its standard deviation        5.61   [ 4.13,  7.07]
         the uncertainty index                     5.04   [ 3.21,  6.87]
         a material: its share at the building 95  3.41   [ 1.21,  5.57]

     **Sixteen of sixteen clear zero.**

     **AND IT ANSWERS DECISION 166, WHICH CALLED THIS THE MOST VALUABLE
     EXPERIMENT LEFT.** That decision recorded a fit threshold of about 81
     declarations becoming a claim threshold near 1,000, a factor of more than
     ten, and named the mechanism: a probabilistic LCA picks ONE method for all
     four of its materials, so one material's fit advantage is averaged against
     three neighbors drawn at random. **Under a per-material policy there is
     nothing to average against and the attenuation very largely disappears**:
     the same rule is worth **+14.2 percent [12.4, 16.1] on the fit** and a
     median **+11.3 percent on the claims**. The hypothesis decision 166 stated
     is confirmed in the direction it predicted.

     **THE FIT HALF IS A CONFIRMATION AND NOT A NEW RESULT.** Its 42.98 percent
     cost over the per-dataset oracle reproduces the minimum of the policy
     curve Stages 2f and 2h already published, to five significant figures.
     What is new is everything at the claim level. **And the threshold was
     calibrated on that fit curve, so the fit number is an in-sample optimum --
     a weak one, since 68 to 106 are indistinguishable -- while the claim-level
     numbers are not: the threshold was never tuned on them.**

     **WHERE THE GAIN COMES FROM, which is the group-composition measurement
     the stage was told to make.** Pooled relative error over every claim,
     split by how many of a group's four materials the rule moves:

         materials above     groups   Lognormal,   KDE,    size   gain over
         the threshold                  uniform   market   rule   best fixed
         0 of 4                 119       32.70    36.82  32.70       0.00
         1 of 4                 573       28.87    31.60  27.31       5.39
         2 of 4                 949       25.17    24.62  21.72      11.78
         3 of 4                 687       21.01    16.57  15.00       9.49
         4 of 4                 172       16.17     8.40   8.40       0.00

     The two zeros are exact and are the control. **Everything the rule buys is
     in the middle and peaks where two of four materials move, which is also
     the commonest composition.** Split instead by the smallest dataset in the
     group: 11.2 percent where it holds 3 to 9 declarations, 10.2 at 10 to 99,
     and exactly 0.00 at 100 to 999, because a group whose smallest material
     clears 100 has all four above the threshold.

205. **2026-09-25, Stage 2j. FOUR QUALIFICATIONS ON DECISION 204, and the
     sharpest is that the rule's ABOVE-threshold choice is wrong on the LEVEL
     claims in the one configuration where the rule does nothing else.**
     `[DELEGATED, 2j measured]`

     **ONE. In the 172 groups of 2,500 where all four materials clear the
     threshold, the rule is the kernel estimate with market weights, and the
     market-weighted LOGNORMAL is closer on nine of the fifteen claims that
     split can score** -- by 39.5 percent [-59.8, -21.2] on the building
     total's mean, 23.0 on a material's share at the building's 95th
     percentile and 20.1 on its mean contribution, three intervals that exclude
     zero. The six claims the kernel estimate wins are SHAPE and FREQUENCY
     claims -- a spread, a rank frequency, how often and how well a
     specification cap works -- and on all six the gain is exactly 0.00 with an
     interval of exactly [0.00, 0.00], the control firing again. **The mirror
     does not hold**: in the 119 groups entirely below the threshold the rule
     IS the best fixed policy on 11 of 15 claims and its largest deficit on the
     other four is 4.7 percent, with no interval excluding zero.

     Two things bound it. The cell is 6.9 percent of a corpus that allocates
     datasets equally across four size bands; on the real EC3 arm, where 31
     percent of categories hold 100 declarations or more, four materials all
     clearing the threshold would happen in about one building in a hundred.
     And POOLED over all sixteen claims the rule still ties the best fixed
     policy there rather than losing. **Fixing it would need a second number in
     the rule, which decisions 88 and 139 have refused twice.**

     **TWO. On the ARGMAX the rule is third, while on the continuous version of
     the same question it is first.** Asked which material is the largest
     contributor it names the truth's answer 48.9 percent of the time against
     52.5 for the kernel with market weights and 50.7 for the lognormal with
     market weights; on the mean absolute error in a material's chance of being
     largest it is 14.3 percent better than the best fixed policy. **This is the
     fourth time this project has found an argmax reading disagreeing with its
     own continuous quantity** -- decisions 102, 105 and 143 are the others --
     and the paper must say which it means every time.

     **THREE. The rule is the most accurate policy per decision and NOT the
     least biased at building scale.** Its absolute error on one building's
     mean total is 8.48 percent of the true total, the lowest of the seven, and
     its signed bias is **-2.85 percent** against the kernel with market
     weights at **+0.92** and the kernel with uniform weights at -1.58. A blend
     of a family that runs low and one that runs near zero inherits a middling
     bias, and decision 122b is why that matters: bias adds across the
     materials of a building while the random part falls as one over the square
     root of the count. **For one building, follow the rule; for a portfolio or
     a stock model, a kernel estimate with market weights everywhere is safer.**

     **FOUR. It captures about a fifth of what a per-material choice could
     buy.** Against a per-material ORACLE -- whichever of the six fixed policies
     is closest on each unit, which needs the answer in order to choose -- the
     rule closes a median 22.8 percent of the distance, range 7.3 to 32.1
     percent. **The oracle is a minimum over six correlated noisy errors and is
     optimistic by construction**, so it is a floor on what is left rather than
     a target; a less optimistic ceiling would need a cross-fitted version and
     is not owned by any stage.

206. **2026-09-25, Stage 2j. A STAGE THAT ADDS A POLICY RUNS ITS OWN TRUTH PASS
     RATHER THAN EXTENDING THE STUDY'S, because a win share, a `best_method`
     and a `stakes` are properties of the SET of policies compared.**
     `[DELEGATED, 2j chose]` Recorded because it cost about sixteen minutes of
     run time that the stage prompt expected not to be spent, and because a
     later stage adding an eighth policy should make the same choice.

     Adding the seventh policy to the existing truth run would have changed
     every win-share denominator, every `best_method` and every `stakes` in the
     six-method tables the paper reports, for a reason that has nothing to do
     with any method changing. The Stage 2j cells therefore sit at the END of
     notebook 3, consume no randomness before any existing cell, and write
     their own tables. **The result is that re-running notebook 3 end to end
     reproduced every pre-existing table CONTENT-IDENTICALLY** -- the only
     differing bytes in the whole of `outputs/` are gzip header timestamps and
     one `written_utc` field -- which is the strongest control this stage has
     that it changed nothing it did not mean to.

     **AND THE STAGE FOUND A DEFECT IN ITS OWN FIRST RUN, by a control that
     should have read zero and read 1.96 percent.** The composition split joins
     each error row to the pLCA group it came from, on the cluster id; the
     design comparison's cluster is a design PAIR from its own resampling whose
     ids run over the same integers, so every pair was handed the composition
     of the same-numbered group. `claim_errors` now records `cluster_kind` and
     `attach_composition` refuses any other kind. **The per-material claims were
     never affected**, because their clusters really are pLCA groups; what was
     wrong was the pooled table, which pools all sixteen claims.

     **The lesson is the one about controls.** The defect was invisible in every
     aggregate and visible only in a cell whose correct value was known in
     advance to be zero. The endpoints of that split were added FOR the control,
     before any number was looked at, and that is what caught it.

207. **2026-09-27, Stage 2j review. FOUR ITEMS THE AUTHOR ASKED TO SETTLE,
     SETTLED.** `[AUTHOR ASKED, THREE SETTLED HERE, ONE CLOSED AS ALREADY
     DONE]` Raised while reading the Stage 2j report's open-items table, whose
     first defect was listing closed items under a heading that says open.

     **ONE. THE TWO ERROR DEFINITIONS ARE SETTLED: the paper's default is the
     PER-UNIT form everywhere, and the portfolio form appears only where the
     sentence is explicitly about many buildings.** The two, both already in
     `TABLE_MetricClaimScorecard.csv` and in this stage's twin of it:

         total_error      mean |error| per unit, over the mean true level.
                          The error in ONE decision -- one building, one design
                          comparison, one specification cap.
         portfolio_error  |mean signed error| over the same divisor. The error
                          in the AVERAGE claim over many decisions, which is
                          what a stock model or a portfolio wants.

     The default is the first because the paper's reader is doing one
     building's LCA. **The second may never be quoted as "the method is
     right"**: on a share or a rank frequency the four materials sum to one, so
     the signed errors cancel exactly and `portfolio_error` is zero by
     construction however wrong each number is. Stage 2h's decision 174 is the
     correction that created the pair; this decides which one the prose means,
     which decision 174 explicitly left open and which was "decided nowhere but
     in this file" for two stages.

     **TWO. THE MODALITY ITEM IS CLOSED AND WAS ALREADY FIXED.**
     `modality_index_fitted` -- the measure decision 134 established and
     decision 82 says the paper should report -- **is present on both arms and
     in every characteristic table**: the corpus's `metrics.parquet`,
     `TABLE_EmpiricalECCMetrics.xlsx` and `TABLE_SyntheticECCMetricsAndW1.xlsx`
     all carry it and its unweighted twin. What it is absent from is
     `coverage.CORE_METRICS`, the nine-characteristic list the coverage table,
     the effective dimension and the CALIBRATION OBJECTIVE are computed over.

     **It stays absent from that list, deliberately.** Adding a tenth
     characteristic would move the calibration objective that decisions 197 and
     200 quote and that the closed generation was judged on, for no benefit now
     that generation is closed. **And the arms agree on it anyway**: the
     standardized arm-to-arm Wasserstein distance is **0.2373** for
     `modality_index_fitted` against **0.2540** for `crit_bw_1`, which IS in
     the list, with empirical and synthetic medians of 1.0258 and 1.0253. So
     the paper can report it from the tables it is already in, and the
     nine-characteristic list stays frozen at what the generator was tuned
     against.

     **THREE. THE CORPUS'S JOINT MODALITY-AND-DISPERSION STRUCTURE IS NOT AN
     OPEN TASK AND GENERATION IS NOT REOPENING.** Decision 203 settled it: the
     corpus keeps `mode_share_alpha = 10` and the shortfall becomes a stated
     limitation, with the numbers on both sides recorded there. The Stage 2j
     report carried it forward as "still open", which reads as work outstanding
     when what is outstanding is a paragraph in the manuscript. It is a
     LIMITATION TO STATE, not a task, and it leaves the open list.

     **FOUR. THE REAL-BUILDING ANCHOR IS CLOSED AND SHOULD NOT HAVE BEEN
     LISTED.** Stage 2i was closed by decision; the anchor is the citation to
     Marsh, Lewis, Hattam and Allen (in press). The row is removed.

     **AND THE FORMAT DEFECT BEHIND ALL FOUR.** A table headed "what is still
     open" listed eight items of which five were closed. **A stage report's
     open list contains open items only**; what a previous stage closed lives
     in this decision log, which is where a reader who wants the history looks.
     The stage report specification in this file is amended to say so.

208. **2026-09-27, Stage 2j review. THE CUTOFF IS SWEPT AND THE PAPER PUBLISHES
     A RANGE. Best 70, indistinguishable 50 to 81, and the whole sweep from 20
     to 220 is worth a fifth of what the rule itself is worth.** `[AUTHOR]`
     "We should publish a range rather than a specific value ... we made
     assumptions, so we shouldn't claim 81 is a precisely correct cutoff
     value."

     Thirteen cutoffs, each scored on all sixteen claims against the true
     parents, on the same instrument `metricreduction.threshold_interval` uses
     at the fit level so the two are comparable: two bootstraps over pLCA
     groups, the second PAIRED against whichever cutoff won on that resample.

         cutoff    20    30    40    50    60    70    81    90   100   110   130   160   220
         pooled  .2073 .2051 .2037 .2028 .2029 .2025 .2026 .2029 .2028 .2031 .2037 .2047 .2058

     **Across the whole sweep the pooled error moves 0.47 points on a level of
     about 20, against 2.68 points for the rule over the best fixed method.**
     The formal indistinguishable run is 50 to 81; 90 falls out on a jitter of
     0.0001 while 100 and 110 come back in, which is the failure mode decision
     142 records, and everything from 40 to 130 sits within 0.0011 of the best.
     **The fit-level sweep agrees independently**: its minimum is at 81 and its
     cost over the per-dataset oracle moves only from 44.0 to 43.0 percent
     across 50 to 100.

     **So the range to print is roughly 50 to 100 declarations.** Getting the
     number exactly right is worth about a fifth of what having the rule at all
     is worth, and decision 142's fit-level threshold of 81 sits inside it.

209. **2026-09-27, Stage 2j review. THE RULE'S GAIN IS THE WEIGHTING SWITCH AND
     NOT THE FAMILY SWITCH, and that reframes how the recommendation should be
     written.** `[AUTHOR ASKED FOR OTHER RULES, MEASUREMENT ANSWERED]` "Are
     there any other rules we should apply to see how they perform?"

     The rule switches two things at one cutoff. Four one-axis variants hold
     one and switch the other, pooled over the sixteen claims:

         the rule: kernel + market above, lognormal + uniform below   0.2026
         kernel throughout, weighting switches at the cutoff          0.2092
         lognormal throughout, weighting switches at the cutoff       0.2096
         best fixed method (kernel, market weights)                   0.2293
         market weights throughout, family switches at the cutoff     0.2295
         uniform weights throughout, family switches at the cutoff    0.2325

     **Switching only the WEIGHTING recovers three quarters of the rule's gain.
     Switching only the FAMILY recovers NOTHING** -- 0.2295 against the best
     fixed method's 0.2293. The family switch is worth a further 3 percent on
     top of the weighting switch, not the other way round.

     **THE AUTHOR'S QUESTION IS ANSWERED IN THE SAME TABLE.** "Are we sure
     lognormal uniform does better? ... What if we try lognormal variable below
     81?" Market weights below the cutoff is `market weights throughout`, and
     it is **13.3 percent worse** than the rule and no better than always using
     a kernel estimate.

     **AND THE REASON IS NOT THAT MARKET SHARE DOES NOT MATTER.** Share of
     datasets on which the market-weighted fit is closer to the truth than its
     own uniform-weighted twin: lognormal 32.8, 42.5, **53.6**, 64.6, 78.7
     percent across 3-9, 10-80, 81-99, 100-999 and 1000+ declarations; kernel
     38.5, 44.1, **52.7**, 59.0, 76.2. **It crosses half at the cutoff, for
     both families, with nothing tuned to make it do so.** And from the oracle
     run, mean absolute error in a material's estimated contribution with
     shares ignored, guessed and KNOWN: kernel 0.1731 / 0.1581 / 0.1378,
     lognormal 0.1670 / 0.1469 / 0.1267. **Knowing beats ignoring everywhere.**
     What loses below the cutoff is the flat-Dirichlet stand-in for shares
     nobody publishes, which is decision 160's finding arriving at the policy
     level. The rule's value is knowing WHEN it is worth guessing.

     **The paper may state the recommendation either as one two-part rule or as
     a weighting switch with a family switch on top.** The second is closer to
     what the measurement says; the wording is the author's.

210. **2026-09-27, Stage 2j review. THE ARGMAX QUALIFICATION IS DROPPED FROM
     THE REPORT AND FROM THE PAPER.** `[AUTHOR]` "Why do we care about the
     argmax? Didn't we agree that probabilistic LCA is complicated and
     distilling it down to that one single metric isn't appropriate?"

     Correct, and the Stage 2j report should not have carried it as a
     qualification. Decisions 102, 143 and 155 demoted the argmax reading three
     times over: it is fragile with four exchangeable materials, it carries a
     3.67 percent noise floor, and it recovers worst of seven candidate
     metrics. Reintroducing it as a mark against the rule gave a retired metric
     a vote.

     The measurement stands in `TABLE_MixedPolicyTruth.csv.gz` for anyone who
     asks -- the rule names the true largest contributor 48.9 percent of the
     time against 52.5 for a kernel estimate with market weights, while on the
     CONTINUOUS version of the same question it is 14.3 percent better than the
     best fixed method -- and it is not a qualification the paper carries.

211. **2026-09-27, Stage 2j review. STAGE 3 ADDS THE RULE AS A SEVENTH COLUMN
     TO THE SCORECARD FIGURE.** `[AUTHOR]` "Yes, Stage 3 should add it as a
     seventh column to the scorecard figure."

     **It is not free and Stage 3 must budget for it.** `best_method`,
     `stakes` and `excess` are properties of the SET of policies compared, so
     the rule wins all sixteen rows and every one of those three columns moves.
     **Every sentence the paper currently writes about which of six methods is
     best has to be re-read against the seven-policy table**, and the
     six-method table is not superseded: it remains the study's comparison of
     METHODS, where the seven-policy one answers whether a per-material POLICY
     beats any fixed method. Quoting a number from one as though it came from
     the other is the mixing decisions 157 and 174 already had to correct
     twice.

212. **2026-09-29, Stage 2j review. "THE STUDY GUESSES MARKET SHARES FROM A
     FLAT DIRICHLET" IS FALSE ON THE SYNTHETIC ARM AND HAS BEEN REPEATED SINCE
     STAGE 2e. The variable arm carries the TRUE market share of every product
     group, exactly.** `[AUTHOR]` "Nobody would ever use the flat Dirichlet
     distribution to guess at market weights ... The two options are applying
     the UQ methods with the market weights known, or applying them with
     uniform weights applied to each value, and seeing how much that costs
     you."

     **The author is right and the code says so.** At the shipped
     `mode_coupling = 1.0`, `generator.draw_weights` gives point i the weight
     `market[group(i)] * within_i`, where `market` is the true market share the
     parent was built with and `within` is a flat Dirichlet INSIDE a group.
     Checked directly on the shipped corpus: the weight mass sitting on each
     product group equals that group's true market share to **1.1e-16**.
     `tests/test_mixedpolicy.py::test_the_synthetic_market_weights_carry_the_true_group_shares`
     pins it against the generator rather than against a stored file.

     **SO THE SYNTHETIC COMPARISON IS IGNORING A KNOWN MARKET SHARE AGAINST
     USING IT**, which is the question the author has been asking for three
     stages. It is not a guess, nobody is proposing one, and describing it as
     one made the result look absurd -- which is exactly how the author read
     it.

     **WHAT THE ORACLE ARM ACTUALLY IS**, and it is much narrower than
     decisions 121 and 160 say. `weighting.oracle_weights` keeps the same true
     group-level share and divides it EVENLY inside a group instead of at
     random. So oracle against variable is not knowing against guessing; it is
     one arbitrary within-group division against another. **Decision 79 already
     said this** -- "the within-mode split of a mode's market share is
     uninformative BY CONSTRUCTION ... finding that it carries none is not
     evidence" -- and nothing carried it forward. **Decisions 121 and 160 are
     NARROWED**: their numbers stand, their "guessed" and "Dirichlet-drawn"
     labels do not, and the oracle column should not be quoted as "known market
     shares" against a "guessed" variable column.

     **THE EMPIRICAL ARM IS DIFFERENT AND THE PAPER MUST NOT BLUR THEM.** Real
     EC3 categories have no published market shares, so
     `empirical.prepare` simulates them with `weighting.coherent_weights` at
     `rho = 0.5`. That arm can say what weighting WOULD do under a plausible
     share model; it cannot say what ignoring a known share costs. **Every
     Stage 2j number is synthetic, so Stage 2j does answer the author's
     question.**

     **AND THE MECHANISM IS THE EFFECTIVE SAMPLE SIZE, not a bad guess.** Using
     a known market share HURTS below about 81 declarations: the market-weighted
     fit is closer to the truth on 32.8 percent of datasets at 3 to 9
     declarations for the lognormal and 38.5 for the kernel, crossing half at
     53.6 and 52.7 in the 81-to-99 band and reaching 78.7 and 76.2 above a
     thousand. Under those true weights the Kish effective sample size has a
     median of **2.8 at 3 to 9 declarations, with 92.4 percent of those
     datasets left below five effective observations**, against a median of
     33.5 at 81 to 99. **A concentrated market share spends your sample**, and
     below the cutoff that costs more than the information is worth. That is
     decision 140's finding stated the right way round.

     **NO NUMBER MOVES.** Every measurement stands; what was wrong is the words
     around it. The manuscript must carry the corrected framing, because the
     old one invites exactly the objection the author raised.

213. **2026-09-29, Stage 2j review. THE CUTOFF SWEEP IS WIDENED TO BOTH
     DEGENERATE ENDS, WHICH MAKES IT SELF-CHECKING, AND ALMOST ANY CUTOFF
     BEATS BOTH FIXED METHODS. This SUPERSEDES decision 208's range.**
     `[AUTHOR]` "Do you mean to say that even if I put the cutoff at 20 or 200
     for KDE vs Lognormal, it performs better than always KDE or always
     lognormal? That's a pretty wild finding. Should we do a wider sweep?"

     Yes, and yes. Twenty-two cutoffs from 3 to 10,000 at 6,000 bootstrap
     resamples, against decision 208's thirteen from 20 to 220 at 2,000.

     **THE SWEEP IS NOW SELF-CHECKING, which is the reason to run it to the
     ends.** The corpus holds 3 to 9,999 declarations, so a cutoff of 3 assigns
     every dataset the kernel estimate with market weights and a cutoff of
     10,000 assigns every dataset the three-parameter lognormal with uniform
     weights. **Both reproduce those fixed methods to 0.00e+00** -- 22.9334 and
     23.9635 -- and the notebook prints that check on every run. A sweep whose
     ends do not land on the methods they are defined to equal is wrong, and
     the narrow version could not show it.

         cutoff     3     5    10    20    40    50    60    70    81    90   100
         pooled  .2293 .2221 .2131 .2073 .2037 .2028 .2029 .2025 .2026 .2029 .2028
         cutoff   130   220   300   500  1000  3000 10000
         pooled  .2037 .2058 .2075 .2109 .2161 .2263 .2396

     **EVERY CUTOFF FROM 5 TO 3,000 BEATS BOTH FIXED METHODS**, 20 of the 22,
     and the only two that do not are the ends, which ARE those methods rather
     than losing to them. **The indistinguishable run is 50 to 81**, with the
     minimum at 70, where decision 208 reported 50 to 81 from a narrower sweep
     and this report recommended rounding to "roughly 50 to 100". **At 6,000
     resamples 90 is excluded and 100 is individually included, so the unbroken
     run stops at 81 and the measured range to print is 50 to 81.** Rounding it
     is an author decision; the measurement is not.

     Across the whole sweep the cutoff moves the pooled error by 3.71 points,
     of which **2.68 is the rule beating the best fixed method** and the rest
     is where the cutoff sits. **So the recommendation is robust in a way a
     single number cannot convey**: the paper should say that switching method
     by size beats any fixed method over two and a half orders of magnitude of
     cutoff, and that the best place to put it is around fifty to eighty
     declarations.

214. **2026-09-29, Stage 2j review. THE STAGE FIGURE IS ONE PANEL. The
     per-claim gains move into the scorecard figure, which Stage 3 owns.**
     `[AUTHOR]` "The figure on the left is fine, but I think that information
     is still better folded into that other scorecard figure that just shows
     how accurate each one is. This figure only makes sense relative to that
     figure ... the other one is more comprehensive and communicates better
     information."

     The two-panel version put the sixteen per-claim gains beside the cutoff
     curve. The gains are a comparison against the best fixed method, so they
     cannot be read without knowing how good that method is, which is what the
     scorecard shows and this figure did not. **They belong in the scorecard,
     as the seventh column decision 211 already asked for**, and the cutoff
     curve stands alone because it says something the scorecard cannot: how
     much the choice of cutoff is worth at all.

     **This also settles a figure-count question the author raised in the same
     message** -- "we're going to have a hard time picking which figures to put
     in the manuscript". Stage 2j contributes ONE figure, the cutoff curve, and
     one column to an existing one.

215. **2026-09-29, Stage 2j review. WEIGHTS DO NOT ADD INFORMATION, THEY RE-AIM
     IT, AND THE EFFECTIVE SAMPLE SIZE IN THE BANDWIDTH IS NOT WHY MARKET
     WEIGHTING LOSES AT SMALL n.** `[AUTHOR ASKED, MEASURED]` "It's confusing
     to me that effective sample size reduces with variable weights. Doesn't
     having values with weights mean you should act like you have more
     information, not less? ... Seems like using a different effective sample
     size might be hurting us there."

     **WHY WEIGHTS DO NOT ADD INFORMATION HERE.** These are not FREQUENCY
     weights, where a row stands for many units you observed. They are
     IMPORTANCE weights on a fixed set of n observed products: you hold n
     EPDs, full stop, and the weights say which of them matter for the target,
     not how many you saw. Market weighting is an importance-weighted
     estimator -- you sampled from the population of products that PUBLISH
     declarations and you want the population that gets BUILT -- so reweighting
     removes the bias and pays variance for it. **The Kish effective sample
     size is exactly that variance cost**, and it is the standard measure of
     it.

     **The concrete case.** Nine EPDs; one product group holds 90 percent of
     the market and contributed two of the nine. The market-weighted estimate
     of that category's distribution then rests on two observations. Knowing
     the share perfectly does not give you more observations of the thing that
     carries the weight.

     **SO IT IS A BIAS-VARIANCE TRADE, the same shape as the kernel estimate
     against the lognormal (decision 74).** Market weighting is unbiased for
     the market-weighted population and high variance; uniform weighting is
     biased -- it estimates a different population -- and low variance. Below
     about 81 declarations the variance dominates and the biased estimator
     wins; above it the bias dominates and the unbiased one wins.

     **AND THE BANDWIDTH IS NOT THE CAUSE, by two independent arguments.**
     First and decisively: **the same crossover appears in the three-parameter
     LOGNORMAL, which has no bandwidth at all** -- market weights are closer on
     32.8 percent of datasets at 3 to 9 declarations and 78.7 percent above a
     thousand. A bandwidth rule cannot cause a crossover in a method that does
     not use one. Second, measured directly in `audits/bandwidth_neff.py` over
     2,000 synthetic datasets, refitting the kernel estimate with the plain
     count in place of the effective sample size:

         declarations      3-9   10-80   81-99  100-999   1000+
         market closer, n_eff   40.1    46.2    61.4     61.6    77.2
         market closer, n       38.9    46.4    56.8     62.4    77.4

     **The crossover does not move.** The uniform-weighted column is
     bit-identical between the two, which is the internal control: with equal
     weights the effective sample size IS the count.

     **The bandwidth rule is unchanged.** Nothing here reopens decision 54 or
     80.

216. **2026-09-29, Stage 2j review. WEIGHTING IS NOT A CHOICE A PRACTITIONER
     MAKES, SO THE RULE THE PAPER RECOMMENDS SWITCHES THE FAMILY ONLY. This
     NARROWS decision 209 and supersedes the recommendation shape of decisions
     139 and 204.** `[AUTHOR]` "A weighting switch isn't feasible. Nobody will
     ever know weights like that. We're just trying to see how much it costs
     the probabilistic fit to assume a uniform fit (which is the only realistic
     option available)." And separately: "considering or not considering weight
     isn't really an option for users, but it's good to know when accounting
     for weight makes a difference."

     **The rule decisions 139, 142 and 204 carry -- kernel estimate with MARKET
     weights above the cutoff, three-parameter lognormal with uniform weights
     below -- is not implementable.** Its upper half needs market shares, and
     on the synthetic arm those are the TRUE shares (decision 212). So it is a
     value of information, not a method.

     **THE STAGE NOW SWEEPS TWO RULE FAMILIES over the same cutoffs.**

         feasible   uniform weights throughout, the FAMILY switches at the
                    cutoff. A reader can follow it with a set of EPDs and
                    nothing else. `mixedpolicy.feasible_policies`.
         known      the same family switch plus the true market shares above
                    the cutoff. The gap between the two curves is what knowing
                    market share would be worth.

     **Decision 209's decomposition is not withdrawn and is re-read.** It found
     that switching only the weighting recovers three quarters of the
     known-share rule's gain and switching only the family recovers nothing.
     That is true AGAINST THE BEST OF ALL SIX fixed methods, which includes
     market-weighted ones a practitioner cannot use. **Against the three
     uniform-weighted methods -- the only ones on offer -- the family switch is
     what there is**, and the stage measures what it is worth on its own.

     **The manuscript's framing follows.** Uniform weighting is not a
     recommendation the paper makes; it is the situation every reader is in.
     What the weighting arm contributes is the SIZE of what that costs, and
     the answer to when it matters: below about 81 declarations, essentially
     nothing, because a concentrated market share would spend the sample
     anyway.

217. **2026-09-29, Stage 2j review. THE RULE A PRACTITIONER CAN FOLLOW IS WORTH
     ABOUT ONE PERCENT, NOT ELEVEN. Decision 204's headline belongs to a rule
     that needs market shares nobody has, and this is the correction.**
     `[DELEGATED, 2j measured after decision 216]`

     Both rules scored claim by claim against the best fixed method a reader of
     THAT rule could otherwise use, 2,500 pLCA groups, paired cluster bootstrap:

         rule        comparator            beats   median gain   range
         feasible    the three uniform-     9/16      +0.9 pct   -0.6 to +6.4
                     weighted methods
         known       all six                16/16    +11.4 pct   +3.3 to +16.8

     **Four of the feasible rule's seven losses have intervals excluding zero**,
     all between -0.25 and -0.63 percent: a material's mean contribution, its
     chance of being largest, its share of the total, and what using 25 percent
     less delivers. **The gains are concentrated in the tail and intervention
     claims** -- the chance of meeting a budget at +6.4, a cap's chance of
     saving 5 percent at +2.8, a material's 95th percentile at +2.7 -- and the
     losses in the mean-and-share claims, where a lognormal fitted to
     everything is already about as good as anything.

     **Pooled it is +2.9 percent** over the best uniform-weighted method, which
     is larger than the per-claim median because the pooled comparison uses one
     comparator throughout while the per-claim one uses the best for each
     claim.

     **AND THE GAP BETWEEN THE TWO RULES IS THE VALUE OF MARKET-SHARE DATA:
     12.8 percent of the pooled error, four times what the feasible rule
     itself is worth.** That is the paper's strongest practical statement and
     it is an argument for obtaining production volumes -- the Marsh, Hattam
     and Allen (2025) route, or an industry-average declaration (decision 100)
     -- rather than an argument about which curve to fit.

     **DECISION 204 IS NARROWED, NOT WITHDRAWN.** Its measurement stands and its
     16-of-16 is real; what was wrong is calling it the recommendation. Decision
     209's decomposition is re-read the same way: switching only the weighting
     recovers three quarters of the known-share rule's gain, and a reader cannot
     switch the weighting.

218. **2026-09-29, Stage 2j review. THE CUTOFF RANGE IS PUBLISHED ROUNDED, AND
     BOTH RULES AGREE ON ABOUT 50 TO 100. This SUPERSEDES the 50-to-81 of
     decision 213.** `[AUTHOR]` "Seems like an oddly specific number ... I
     think that might be more significant digits than we can promise. I think
     we should publish a rounded range."

     The author is right and 81 was never a claim-level number: it is decision
     142's argmin of a FIT-level curve on a dense grid, carried forward because
     the study had it. On the claim-level sweep at 6,000 bootstrap resamples:

         rule        best cutoff   indistinguishable   whole sweep spans
         feasible        130           50 to 130          0.73 points
         known            70           50 to 100          3.71 points

     **BOTH BANDS MOVED WHEN DECISION 224 REPLACED THE BAND RULE**, which until
     then measured each cutoff against whichever cutoff won its own resample --
     a contest among grid points rather than a difference test. On the fixed
     rule the feasible band is **40 to 170** and the known-share band is **50
     to 110**. The published range is the feasible rule's, by decision 220, and
     is unaffected by the known-share figure.

     Decision 213's 50-to-81 came from a grid that included 81 as a legacy
     point and 90 as a neighbor that fell out on jitter. On the rounded grid
     the unbroken run and the individually-indistinguishable span agree for
     both families, and both contain **50 to 100**. **That is the range to
     print**, and the feasible rule tolerates up to 130.

     `threshold_curve` now reports the indistinguishable span beside the
     unbroken run, because the run rule was written to stop a lone far-away
     point widening a band (decision 142) and is the wrong instrument for a
     hole in the middle.

219. **2026-09-29, Stage 2j review. THE BANDWIDTH IS SETTLED, AND THE
     EXPLANATION THAT WAS GIVEN FOR THE CROSSOVER WAS SLOPPY. What ignoring a
     known market share costs is a BIAS-VARIANCE TRADE WITH A FLOOR, and the
     uniform-weighted fit has an error it can never get below.** `[AUTHOR
     PRESSED, MEASURED]` "You have nine observations. Then you have additional
     information about each of those observations. So it doesn't make sense
     that n_effective would be less than nine ... I bet if you used a regular
     n, KDE variable would dominate."

     **THE HYPOTHESIS IS TESTED AND IT IS WRONG, three ways.** Share of
     datasets on which the market-weighted kernel estimate is closer to the
     true market-weighted parent than its own uniform-weighted twin, 2,500
     synthetic datasets, `audits/bandwidth_neff.py`:

         declarations                        3-9   10-80   81-99  100-999   1000+
         production rule, n_eff             40.3    44.0    53.8     61.4    77.9
         the plain count n                  38.8    43.7    50.0     61.9    78.4
         each fit's OWN BEST bandwidth      46.0    42.7    51.9     55.3    74.6

     **The plain count does not flip it. And giving each fit the bandwidth that
     minimizes its own distance to the truth -- which no rule can beat --
     does not flip it either.** The bandwidth costs the market-weighted fit
     about 6 points at 3 to 9 declarations and the crossover is still below 81
     without it. A fourth argument needs no bandwidth at all: the
     three-parameter lognormal has none and shows the same crossover.

     **THE AUTHOR IS RIGHT THAT KNOWING WEIGHTS TAKES NO INFORMATION AWAY, and
     the earlier wording in the stage report -- "leaves the estimate resting on
     two observations" -- was wrong and is withdrawn.** Nine EPDs are nine
     EPDs. What changes is the QUESTION: nine declarations give nine
     observations of the population that PUBLISHES, and learning the shares
     reveals that seven of them describe a group that is ten percent of what
     gets BUILT. The effective sample size is not a penalty the method imposes;
     it is how many of your nine are aimed at the question you now know you are
     asking, and the sample was always aimed that way.

     **AND THE MECHANISM IS A FLOOR, which is the better result.** Each fit at
     its own best bandwidth, mean W1 against the true market-weighted parent:

         declarations            3-9   10-80   81-99  100-999   1000+
         uniform weights      0.2621  0.1614  0.1142   0.0968  0.0844
         known market shares  0.3005  0.1808  0.1361   0.0762  0.0287
         distance between the two true populations  0.10 to 0.12 at every size

     **The market-weighted fit converges toward zero -- 0.30 to 0.029 and still
     falling. The uniform-weighted fit flattens at 0.084 against a floor of
     0.099**, which is the distance between the population that publishes and
     the population that gets built. **No quantity of EPDs takes a
     uniform-weighted fit below that floor.** Below 81 declarations its
     variance advantage exceeds the floor; above it, it does not. That is the
     ordinary bias-variance trade, the same shape decision 74 records for the
     kernel estimate against the lognormal.

220. **2026-09-29, Stage 2j review. THE PUBLISHED CUTOFF RANGE IS 50 TO 130,
     the feasible rule's own indistinguishable span. This SUPERSEDES the "about
     50 to 100" of decision 218.** `[AUTHOR]` "If 50-130 is indistinguishable,
     we should publish 50-130 as the range."

     Decision 218 rounded to 50-100 because that is where the two rule families
     overlap. The rule the paper recommends is the FEASIBLE one (decision 216),
     and its own measured span is 50 to 130, with the minimum at 130 and the
     whole sweep from 3 to 10,000 worth only 0.73 points. The known-share
     rule's 50-to-100 stays in the tables; it is not the recommendation and
     does not constrain the printed range.

221. **2026-09-29, Stage 2j review. THE WEIGHTING RESULT IS FRAMED AS WHAT NOT
     KNOWING MARKET SHARES COSTS, AND IN ABSOLUTE POINTS.** `[AUTHOR]` "'What
     ignoring a known market share costs' -- this isn't a real scenario. Frame
     it as 'what not knowing a market share costs'." And: "It seems like it
     doesn't end up being that important, which is good."

     Nobody chooses to ignore a share they know, so the heading described a
     situation that does not arise. **And the size of it should be stated in
     absolute points, not only as a ratio.** Giving the rule the true market
     shares above the cutoff takes the pooled error from **23.24 to 20.25
     percent** of the true level: **2.98 points, or 12.8 percent of what was
     there**.

     **The relative number oversells it and the absolute one is the honest
     frame.** A probabilistic LCA is wrong by about 23 percent either way, and
     knowing every market share exactly would take it to 20. The author's read
     -- that weighting does not end up being the big factor -- is what the
     absolute number says, and the paper should lead with that form. It is the
     same distinction decision 156 settled for the scorecard: a spread between
     methods means nothing without the level the best method still gets wrong.

222. **2026-09-29, Stage 2j review. THE CROSSOVER IS IN THE MEAN, NOT IN THE
     SHAPE, AND THE AUTHOR'S OBJECTION IS WHAT FOUND IT. Below about eighty
     declarations, knowing the true market shares makes your estimate of the
     market-weighted MEAN WORSE.** `[AUTHOR PRESSED, MEASURED]` "If we get 2
     EPDs from a large city and 7 from a small town, wouldn't distributions
     that treat all those values equally be a lot worse than the distributions
     that take those differences into account? The parent distribution is
     probably more clustered around the 2 EPDs, so a distribution that accounts
     for that would be better and closer to the parent."

     **THE PREMISE IS RIGHT ABOUT THE TARGET AND THE CONCLUSION NEEDS THE
     SAMPLING ERROR.** The market-weighted parent IS mostly the city's
     distribution. What the objection leaves out is that you hold **two**
     observations of the city.

     **THE INSTRUMENT.** `audits/weighting_location_shape.py` applies decision
     92's split to a fit against a parent rather than to one dataset under two
     weightings. W1 is the area between two CDFs and is bounded below by the
     distance between their means, so

         W1 = integral |F_model - F_parent|
         LOCATION = |integral (F_model - F_parent)| = |mean difference|
         SHAPE    = W1 - LOCATION, non-negative on the grid by construction

     Location is whether the fit is AIMED at the right population; shape is
     whether it KNOWS that population. Two estimators, so no bandwidth rule can
     be the answer either way: the kernel estimate at each fit's OWN BEST
     bandwidth, swept, which nothing can beat; and the three-parameter
     lognormal, which has no bandwidth at all.

     **THE LOCATION TERM, mean over datasets, two independent 2,000-dataset
     draws, kernel estimate at its own best bandwidth:**

         declarations        3-9    10-80   81-99  100-999   1000+
         uniform weights   0.1918  0.1105  0.0742   0.0715  0.0663
                           0.1912  0.1067  0.0695   0.0682  0.0666
         known shares      0.2183  0.1117  0.0874   0.0442  0.0153
                           0.2332  0.1418  0.0611   0.0430  0.0145

     **Knowing the shares makes the MEAN worse below about eighty declarations
     and four times better above a thousand**, and the three-parameter
     lognormal gives the same shape on both seeds. **So the crossover is not
     about distributional shape, not about a kernel, and not about a bandwidth.
     It is in the simplest statistic there is.**

     **THE MECHANISM, in the author's own example.** The market-weighted mean of
     nine EPDs where two carry ninety percent of the weight is arithmetically
     close to the average of two numbers, and it has the standard error of one:
     the Kish effective sample size is 2.8 in the median at three to nine
     declarations. It is aimed at exactly the right quantity and it is wild. The
     unweighted mean of all nine is aimed at the wrong quantity -- the
     population that PUBLISHES rather than the one that gets BUILT -- and every
     observation contributes to it. At nine EPDs the noise in the first exceeds
     the bias in the second. At a thousand, the group carrying the weight has
     hundreds of declarations of its own, the noise is gone, and only the bias
     is left.

     **SHAPE MOVES THE SAME WAY AND IS THE SMALLER TERM**, 0.074 to 0.079 and
     0.080 to 0.078 at three to nine declarations against location's 0.19 to
     0.23. So the trade is decided by the mean, which is the term a reader can
     check by hand.

     **A 300-DATASET RUN OF THIS AUDIT SHOWED THE OPPOSITE SIGN ON LOCATION AND
     IS NOT A RESULT.** It put the market-weighted location below the uniform
     one in every band. Two independent 2,000-dataset draws agree with each
     other and disagree with it. **Nothing from the 300-dataset run is quoted
     anywhere**, and it is recorded because a smaller sample of the same script
     would reproduce it.

     **THE AUTHOR'S OWN STATEMENT OF IT, 2026-09-29, AND IT IS THE SENTENCE
     THE PAPER SHOULD CARRY:** "It's not that market share isn't helpful; it's
     that market share with very few values might lead you to over-index on
     something because it's misleading. Once you have a representative sample,
     market share is a big helper." **Over-index is the right word**: with two
     declarations carrying ninety percent of the weight, the whole estimate is
     indexed on two values that may not represent their own product group.

     **ONE PRECISION TO KEEP WITH IT.** The sample is unrepresentative of the
     market at EVERY size -- that is the whole reason weighting exists, and
     weighting is what corrects it. What changes with dataset size is not
     representativeness but **how many declarations sit inside the group that
     carries the weight**: two at nine EPDs, hundreds at a thousand. The paper
     should write "enough declarations in the products that dominate the
     market" rather than "a representative sample", because the second inverts
     which of the two things weighting is for.

     **WHAT THIS CHANGES.** No number moves and decisions 215 and 219 are
     confirmed rather than narrowed: the trade is bias against variance, the
     bandwidth is not the cause, and the effective sample size is the variance
     cost. **What is replaced is the EXPLANATION.** The stage report said
     learning the shares "reveals that your sample is badly aimed", which is a
     restatement rather than a mechanism and did not answer the objection. The
     measured statement is that the estimate of the market-weighted mean rests
     on however many declarations actually carry the weight, and below about
     eighty that is too few to beat an unweighted average of all of them.

         python audits/weighting_location_shape.py --n 2000

223. **2026-09-30, Stage 2j review. `flip.FLIP_THRESHOLDS` IS 0.0029, 0.015 AND
     0.032, AND FOUR DECISION ENTRIES STILL PRINT THE VALUES IT REPLACED. The
     code is right and the log is stale.** `[DELEGATED, found by the Stage 2j
     review]`

     `src/flip.py` carries `{0.01: 0.0029, 0.05: 0.015, 0.10: 0.032}`. Decisions
     **95, 108, 173 and 175** print **0.0018, 0.011 and 0.025** as the published
     crossings, in seven places, none of which says they were superseded.

     **The code is the current value and it is an AUTHOR decision, not a drift.**
     The recalibration was taken at the close of Stage 2h and is recorded in
     `reports/START_HERE.md` section 5: the three stored values had all fallen
     outside their own recomputed 95 percent intervals -- 0.0018 against a
     recomputed 0.00291, 0.011 against 0.01502, 0.025 against 0.03157, off by
     factors of 1.62, 1.37 and 1.26 -- and notebook 1 was re-run on the new
     constants, moving the mean probability that unknown market shares change
     which material leads from 0.9870 to 0.9734 at the 1 percent level, 0.8642
     to 0.8144 at 5 percent and 0.7105 to 0.6524 at 10 percent.

     **WHY IT MATTERS ENOUGH TO GET ITS OWN ENTRY.** These are PUBLISHED
     CONSTANTS. A manuscript session reading the decision log rather than the
     code would print numbers 30 to 45 percent low, and the log is the thing
     this project tells such a session to read. The four entries now carry a
     superseding note in place; this entry is where the current values live.

     **AND THE PROSE RULE OF DECISION 175 APPLIES TO THE NEW VALUES TOO**, so
     whatever the paper prints must be fitted both ways and rounded at the first
     digit the two fits disagree on. That has not been redone since the
     recalibration and Stage 3 owns it.

         python -c "import sys; sys.path.insert(0,'src'); import flip; print(flip.FLIP_THRESHOLDS)"

224. **2026-09-30, Stage 2j review. THE PUBLISHED CUTOFF RANGE IS 40 TO 170, the
     indistinguishable band, and NOT the argmin's bootstrap interval. And the
     band rule itself was a contest rather than a difference test until this
     entry.** `[AUTHOR]` "If it's indistinguishable 40 to 170, why are you
     recommending 50 to 140? ... I'm not sure how 50-140 and 40-170 are answers
     to different questions. This is a calculation."

     **THE BAND RULE WAS WRONG AND IS FIXED.** `threshold_curve` measured each
     cutoff's penalty as its excess over whichever cutoff won THAT RESAMPLE.
     Subtracting a per-resample minimum makes every penalty non-negative by
     construction, so a cutoff's lower bound could reach zero only by WINNING
     some resamples -- a contest among however many near-tied points share the
     grid, not a test of a difference. Filling the grid from 14 points to 27
     split the wins among more neighbors and moved the reported band while the
     curve itself did not change, and opened a hole at 80 to 90 across a span
     whose pooled error varies by 0.00013 on a level of 0.232. The penalty is
     now the excess over the cutoff that wins on the FULL SAMPLE, held fixed, and
     a cutoff is indistinguishable when that interval straddles zero. The band is
     then **40 to 170, contiguous**, with 30 and 180 out.

     **AND THE ARGMIN INTERVAL IS NOT THE THING TO PUBLISH.** It is [50, 140].
     The author's objection was that it is a min-statistic, of the kind decisions
     102, 143 and 210 demoted. **That specific worry was tested and does not
     hold**: on a curve constructed to be exactly flat from 40 to 170 the argmin
     bootstrap recovers 40 to 170 exactly, with 13.3 percent of argmins landing
     on the two edge points, so it is not pathologically drawn inward. An earlier
     draft of this entry asserted an edge artifact and that assertion is
     WITHDRAWN -- it failed its own test.

     **The real reason is that it answers a different question.** [50, 140] is
     where the single lowest point of the curve lands on resampling; it is
     narrower than the band because the curve genuinely tilts at the band's edges
     -- 40 sits 0.00025 above the minimum and 170 sits 0.00017 above, both inside
     the noise. **A practitioner is not choosing the optimum, they are choosing
     something good enough**, so the band is the quantity that answers their
     question and the argmin interval is a diagnostic.

     **AND THE BAND IS CONSERVATIVE.** Its reference is the argmin of the same
     data, so differences measured against it are biased slightly positive and
     the band is if anything too narrow.

         Reproduce: outputs/tables/TABLE_MixedPolicyThreshold.csv, family=feasible

225. **2026-09-30. THE PAPER QUOTES TWO NUMBERS AND BOTH ARE RANGES: 40 to 170
     declarations for the family split, 80 to 100 for the weighting one. NO
     SINGLE-DECLARATION THRESHOLD IS PUBLISHED, at either level.** `[AUTHOR]`
     "81 is not a number worth quoting at all. Too specific. The only numbers
     we're quoting are 40-170 for KDE/lognormal split and 80-100 for
     market/uniform split. I don't know why I have to keep reiterating this."

     **The author said it three times and each of my answers kept a use for 81
     alive**: first as the claim-level constant, then as the fit-level argmin
     that was "still correct", then as a thing the fit-level text should also
     turn into a range. The decision is simpler than any of those. **Nothing in
     the manuscript prints a single-declaration cutoff.**

         THE FAMILY SPLIT        40 to 170 declarations   decision 224
         THE WEIGHTING SPLIT     80 to 100 declarations   decision 222

     `mixedpolicy.MIXED_THRESHOLD = 80` stays, because the code needs one value
     to name one policy and to define the reference the gain table is computed
     at. **It is a constant, not a result**, and no table, figure, caption or
     sentence presents it as an answer.

     **What this supersedes in presentation only, with no number moving.**
     Decision 142's 81 and decision 198's reproduction of it are measurements
     and stand as the record of what was found; what changes is that the FIT
     level publishes a range too, on the same grounds the author gave for the
     claim level. Decisions 139, 204, 208, 213, 218 and 220 each printed a point
     estimate or a narrower range at some stage; 224 and this entry are the
     final form. **A later stage that finds 81 in a draft should treat it as
     stale rather than as a result to defend.**

226. **2026-10-01. GROUPED MARKET WEIGHTS ARE TESTABLE ON THIS CORPUS AND ARE A
     DISCUSSION POINT FOR THE MANUSCRIPT, NOT A LIMITATION. My claim that the
     generator makes them untestable was wrong, twice over.** `[AUTHOR
     CORRECTED ME]` "Knowing group sums is not equivalent to knowing individual
     shares ... this generator definitely can say what real shares would do.
     Stop treating this like an impossibility."

     **THE PROPOSAL, which is KL2's group weight constraints.** A practitioner
     will never know every product's market share, but may well know the summed
     share of a GROUP -- a production route, a region. Split a category's values
     into 2 to 5 groups, keep each group's summed weight, and assume the weight
     is spread EVENLY inside the group. Nine values where three sum to 60
     percent become 20 percent each, against a truth that might be 2, 48 and 10.

     **MY FIRST ERROR: I said that is equivalent to knowing the shares.** It is
     not, and the author's arithmetic is the refutation -- 20/20/20 is a
     different weight vector from 2/48/10 and produces a different fit. What
     decision 79 actually established is narrower: it is equivalent **when the
     groups are the generator's own mixture components**, because every point in
     a component is drawn from the same density, so the within-group split
     cannot move the target. That is a statement about ONE choice of grouping,
     not about grouping.

     **MY SECOND ERROR: I called the experiment untestable here.** Under the
     author's construction -- groups drawn at RANDOM, not aligned with the
     components -- a group mixes components, averaging inside it genuinely
     destroys information, and the cost is real and measurable. The corpus can
     in fact BRACKET the question, which is better than either bound alone:

         groups = the true components     an upper bound, recovers everything
         groups drawn at random           a lower bound, blurs across components
         2 to 5 groups swept              how fast the gap closes with k

     **WHAT THE MANUSCRIPT OWES, and it is more than a sentence.** The gap
     between the orange and blue curves -- about 3 points of the roughly 23 a
     probabilistic LCA gets wrong -- is what full market shares would buy.
     Grouped weights are a far more realistic information state than full
     shares, and they close some of that gap. The discussion should say so, in
     generalities, name KL2 as where the machinery already exists, and mark the
     measurement as future work rather than claiming a number this stage did not
     produce. **It is an avenue for making the model better, not a limitation of
     the corpus.** Discrepancy entry 185.

     **AND THE GENERAL LESSON ABOUT MY OWN REASONING.** Both errors ran the same
     way: I took a result that holds under one specific construction -- decision
     79's, with groups equal to components -- and restated it as a property of
     the corpus. A claim that something cannot be measured deserves at least as
     much scepticism as a claim that it can, and I reached for it twice without
     testing it once.

227. **2026-10-01, Stage 3. COMPUTE AND PLOTTING ARE SEPARATED IN NOTEBOOKS 1
     AND 2, AND THE SPLIT MOVED NOTHING.** `[AUTHOR]` This is Stage 1's deferred
     Phase 5 and Stage 3's first task, and it had TWO blockers rather than the
     one the prompt file carried.

     `audits/render_figures.py` raises on a missing `OUT` BEFORE it checks
     markers, and neither notebook defined one, so marking their figure cells
     would have unlocked nothing. `tests/test_render_figures.py` skipped both
     with that reason. The second blocker is the one that mattered: their figure
     cells read fitted models and frames out of kernel memory, which the
     standing constraint forbids and which is why those figures could not be
     reproduced without a 30- and a 35-minute run.

     **Where a figure needed something never persisted, the cell is SPLIT**: a
     compute half that writes a table, a figure half that draws it. Eight new
     tables carry what used to live only in memory -- the example parents, the
     example datasets by stratum, the W1 definition demo, the six fitted
     densities of one dataset, the KS and W2 scores, the rank frequencies, and
     every empirical dataset's values and fitted curves.

     **THE SPLIT PRESERVES THE RANDOM STREAM EXACTLY, WHICH IS THE WHOLE POINT.**
     Notebook 1's example figures draw from the main Generator, and
     `corpus.make_combos` later takes the pLCA groupings from that same stream,
     so moving a draw would have moved every pLCA number in the paper. The
     compute halves consume what the single cells consumed, in the same order;
     the figure halves consume nothing.

     **THE CONTROL, and it is as clean as this project has had.** Notebook 1
     re-ran end to end: all 147 x 24 empirical characteristic values are BIT
     IDENTICAL, and four of its five figure PNGs are BYTE IDENTICAL. The fifth
     changed because an axis label said "W1, uniform vs variable" and now says
     "market weights". Notebook 2's synthetic score table is bit identical over
     10,000 rows and 39 columns.

     **The one exception is the strip jitter, and it cannot touch an analysis
     stream.** It comes from a Generator seeded independently from `SEED`,
     because spawning one would renumber every later spawn and move the
     empirical weights and the cross-validation splits.
     `tests/test_notebooks.py` permits exactly that, declared with a marker, and
     still fails on an undeclared second Generator.

         python -m pytest tests/test_render_figures.py tests/test_notebooks.py -q

228. **2026-10-01, Stage 3. FIGURES 2 AND 3 ARE ONE FIGURE, AND THE EMPIRICAL
     STRIP IS NEW.** `[AUTHOR]` `CompareUQMethods_FIG_W1DistanceAndRank`: the W1
     strip with mean labels on the left, the rank-frequency heatmap on the
     right, one row per arm, one cell, one file. The synthetic strip existed and
     the empirical one did not, so the merge adds a panel rather than only
     combining two.

     The two superseded images are in `archive/figures/` with a reason, not
     deleted.

229. **2026-10-01, Stage 3. A FIGURE IS WRITTEN BY ONE CALL THAT TAKES A STEM,
     EMITS A VECTOR SIBLING, AND REFUSES A NAME TWO PLACES WRITE.** `[AUTHOR]`
     `figstyle.savefig(fig, OUT, stem)`. The naming convention lives in one
     place, a journal gets a PDF beside every PNG from the same call so the two
     cannot drift, and a duplicate stem raises.

     **That last part is not hypothetical.** This repository carried two
     different figures both called `FIG2`, which is what happens when two cells
     write one name: nothing fails and the second silently discards the first.

     **`audits/figure_manifest.py` joins every image on disk against the code
     that writes it**, and reports three things: orphans, duplicates, and which
     figure cells the fast renderer can execute on its own. It found FIVE images
     with no generator anywhere; they are in `archive/figures/` with reasons.
     `tests/test_figure_manifest.py` fails on a duplicate filename, on an orphan,
     on a retired weighting word in a filename, and on a PNG with no vector
     sibling.

     **AND IT FOUND THAT NOTEBOOK 3 WAS NEVER RENDERER-SAFE.** 28 of 37 figure
     cells run against the setup block alone; the nine that cannot are all in
     notebook 3, which was believed clear because every use of it had passed
     `--only` and rendered a cell that happened to be self-sufficient. Stage 4
     owns it; `TABLE_FigureRendererSafety.csv` names them.

         python audits/figure_manifest.py

230. **2026-10-01, Stage 3. THE SCORECARD HAS ITS SEVENTH COLUMN, AND IT CARRIES
     TWO COUNTS THAT MUST NOT BE QUOTED FOR EACH OTHER.** `[AUTHOR]` This
     implements decision 211. The column is the FEASIBLE rule -- uniform weights
     throughout, kernel estimate above the cutoff, three-parameter lognormal
     below -- and never the known-share one.

     **The rule is closest of ALL SEVEN on 2 of 16 claims and closest of the
     FOUR A READER CAN CHOOSE on 10 of 16.** Both are true and they answer
     different questions.

     **THE FIGURE IS GROUPED BY WHETHER A READER CAN CHOOSE A COLUMN AT ALL**,
     which is the distinction the paper turns on and which the old ordering
     buried: three uniform-weighted fits and the size rule on the left under
     "what a reader can choose", the three needing market shares on the right
     under "what market shares would buy", a gap between, and the lower panel
     on the same positions with the size rule blank. A solid box marks the best
     of the four, a dashed box the best of all seven. **The title is the solid
     box count** -- "switching family by dataset size is the best available
     choice on 10 of 16 claims", chosen by the author from four options on
     2026-10-01. The margin behind it is 0.7 points pooled, 23.25 against
     23.92, which the cells show and the title does not.

     **NEITHER IS DECISION 217'S 9 OF 16.** That one is the rule's gain over the
     best uniform-weighted method for each claim with a PAIRED interval
     excluding zero; mine is a plain argmin. They differ on exactly one claim, a
     material's standard deviation, where the rule is closest by 0.52 percent on
     an interval running -0.37 to +1.46. Pooled over the sixteen: the rule 23.25
     percent of true level, the kernel estimate with uniform weights 23.92, the
     three-parameter lognormal with uniform weights 23.96.

     **No cutoff is printed anywhere on the figure**, per decisions 224 and 225;
     the footnote gives the range.

     **`metricset.rescore` is the one implementation of the derived columns.**
     `best_method`, `stakes`, `excess` and `methods_differ` are properties of
     the SET of methods compared, so a seventh policy changes all four;
     `claim_scorecard` calls the same function, which is what stops the
     six-method and seven-policy tables disagreeing about how they were derived.

         python -c "import pandas as pd; d=pd.read_csv('outputs/tables/TABLE_ClaimScorecardWithRule.csv'); print(d[d['rank']==1].method.value_counts())"

231. **2026-10-01, Stage 3. DECISION 175'S ROUNDING RULE GAINS ONE GUARD: the
     printed spread may not be more than twice the real one.** `[DELEGATED, 3
     found and fixed]` The rule prints a crossing at the first significant
     figure the logistic and the isotonic fit disagree on. On the recalibrated 5
     percent flip threshold it rendered **0.015021 and 0.013899 -- which agree to
     7.5 percent -- as "0.02 against 0.01"**, which reads as a factor of two. The
     cause is a value sitting on a rounding boundary, where one unit in the last
     place is far larger than the difference being displayed.

     The chosen digit must now also satisfy that the gap between the PRINTED
     numbers is at most twice the gap between the real ones. It only ever
     increases the digit count, and **both of decision 175's published examples
     return exactly what they did**: 2.1 against 2.2, and 1.5 against 1.4.

     **Two prose values move and no constant does**: the 5 percent crossing from
     "0.02 against 0.01" to **"0.015 against 0.014"**, and the 10 percent from
     "0.032 against 0.031" to **"0.0316 against 0.0314"**.

     **AND `CONTEXT.md` WAS STILL PRINTING THE SUPERSEDED CONSTANTS.** Decision
     223 corrected the four decision-log entries that carried 0.0018, 0.011 and
     0.025 and did not reach the mechanics file, which a session is told to read
     before touching code. Corrected, and
     `tests/test_flip.py::test_stored_flip_thresholds_sit_inside_their_own_intervals`
     now fails if the stored constants drift outside their own recomputed
     intervals again -- which is the open item Stage 2j handed here.

232. **2026-10-01, Stage 3. THE TWO HUMP-SPACING LEVERS ARE MEASURED ALONE UNDER
     THE SETTLED WEIGHT RULE, AND THE RECOMMENDATION IS STILL NOT TO
     REGENERATE.** `[DELEGATED, 3 measured]` `separation_dispersion_frac` and
     `shoulder_frac` were both still at 0.0 and every earlier measurement of
     either was taken under the mismatched weight rules, on the superseded
     corpus, or bundled with `mode_share_alpha` so neither could be read on its
     own. 1,000-dataset drafts, judged against the **weight-draw noise of 0.006
     to 0.015**:

         candidate        multimodal  both   multi|dispersed  objective  weighting
         shipped              0.200  0.013            0.147     0.2281     0.1722
         + separation         0.233  0.022            0.191     0.2730     0.1520
         + shoulder           0.167  0.019            0.167     0.2214     0.1420
         the real arm         0.246  0.054            0.212         --         --

     **Separation closes most of the conditional gap** -- 0.147 to 0.191 against
     a real 0.212 -- **and costs 0.045 on the objective, three to seven times the
     noise.** Shoulder is free on the objective at -0.0067, buys about half the
     joint gap, and gives up the multimodal marginal, 0.200 to 0.167.

     **AND BOTH IMPROVE THE WEIGHTING MARGIN**, 0.1722 to 0.1520 and 0.1420,
     where every pre-port measurement had widening cost it. **Decision 193 is
     confirmed on a lever it was never measured on.**

     Neither closes the gap and both have a price, so **decision 203 stands and
     generation stays closed**. What is new is that the limitation now rests on a
     direct measurement of the two levers built for it.

     **A DEFECT FOUND DOING IT, and it is the stale-input failure this project
     keeps hitting.** The audit reused any existing draft directory, and the
     `current` draft had been generated under the SUPERSEDED configuration --
     `cv_log10_mean` 0.129 against 0.329, `min_q1_over_iqr` 0.5 against 0.2 --
     so every candidate was being compared against the wrong baseline while the
     table said `current`. It now refuses a draft whose recorded configuration
     differs from the one asked for, and regenerates under a dated label rather
     than overwriting a corpus.

         python audits/corpus_joint_structure.py 1000 --only current,shipped_plus_separation,shipped_plus_shoulder

233. **2026-10-01, Stage 3. THE SIX STALE AUDITS ARE RE-RUN ON THE CURRENT
     CORPUS. EVERY ORDERING HELD, EVERY LEVEL MOVED, AND THREE DECISION ENTRIES
     NEED THEIR NUMBERS UPDATED.** `[DELEGATED, 3 measured]`

     **Decision 182, the upper truncation.** The worst runaway fit is **4.95**
     times the data's own spread uncapped, not 5.4, and a cap at twice the
     largest observation takes it to **1.85**, not 1.8. Against the truth the cap
     still costs nothing: `Lognormal, Variable` 0.19924 uncapped against 0.19913
     capped, and the kernel estimate and the normal do not move. Decision 199's
     one-line remedy stands with the new figures.

     **Decision 184, the judgment arm.** The six data-driven methods span 0.0756
     to 0.1140 on the design comparison; a correctly centered pedigree model at
     matched spread reads **0.0886**, and 0.0886 to 0.1136 across a six-fold
     spread range. A common offset cancels to the last digit, which reproduces
     that decision's control exactly. The realistic one-declaration case reads
     0.1612 to 0.2722. At matched spread and correct center, uniform reads
     **0.2349** and triangular **0.2228** against the pedigree lognormal's
     0.0886, so the shape finding is **2.5-fold** rather than 2.1-fold.

     **Decision 187, the certification credit.** At a 10 percent credit with 75
     percent confidence the truth earns it on **12.7 percent** of 3,000 cases,
     at least two methods disagree on **19.4**, the best method is wrong on
     **7.5** and the worst on **13.0**. The fragility is still the threshold:
     the six disagree on 59.1 percent of designs within 0.05 of the line and 5.1
     percent beyond 0.25. **A normal is the worst method on 9 of the 15
     tier-and-confidence combinations**, where that decision records 8.

     **Decision 195, the bandwidth through the pLCA, and this one is NARROWED.**
     Averaged over the five outputs Scott is still worst under both weightings
     -- 29.23 against 28.24 and the shipped rule's 28.51 under uniform weights,
     30.06 against 28.89 and 28.93 under market weights -- so the shipped rule
     stands. **What does not survive is "Scott is worst on every one of the five
     outputs":** under uniform weights Scott is now best on two, a material's
     standard deviation at 31.091 against 31.035 and the uncertainty index at
     49.412 against 49.428. **And the guard no longer BUYS anything under market
     weights**, where that decision records it buying 0.15 of a point; it costs
     0.04 there and 0.28 under uniform weights. The control fires: the four
     parametric methods move 0.0000 points across the three rules.

     **Decision 188, the bandwidth sweep**, is unchanged: the two arms disagree
     about the density criterion and agree about the study's own.

234. **2026-10-01, Stage 3. THE TWO OVERSIZED TABLES MOVE TO PARQUET, AND THE
     ASCII AND SPELLING SWEEPS ARE DONE.** `[AUTHOR]`

     `TABLE_MethodCurves` 96.4 MB to **44.4 MB** and `TABLE_MethodWinShare` 22.8
     to **6.7**, with every float still float64, every row verified element by
     element before the CSVs were removed, and a 23-fold speedup on read.
     Decision 15 already chose Parquet for this repository's bulk data and gave
     the reason that applies here: a Zenodo reader can open it from R or Julia.

     **THE UNICODE MINUS WAS STILL IN MOST FIGURES.** `figstyle.apply()` has set
     `axes.unicode_minus` since Stage 2e, and the new style audit shows **30 of
     36 marked figure cells never call it**, so every figure with a negative tick
     label carried U+2212. Each notebook's matplotlib config cell now sets it,
     which reaches all of them without re-tuning layouts built before the style
     module existed. **The rest of FIGURE_STYLE.md compliance is a redesign of 30
     figures and is NOT done**; `TABLE_FigureStyleCompliance.csv` names them.

     **US spelling across 19 files.** Two exclusions, both deliberate: the sent
     prompt sections are a record of what a session was given, and
     "characterization" appears only inside the title of Marsh, Lewis, Hattam
     and Allen (in press), which keeps its published spelling. **The sweep broke
     two things and both are repaired**: a local variable in
     `metricreduction.py` whose assignment was skipped by the identifier guard
     while its use was renamed, and a regex in `audits/judgment_arm.py` parsing a
     label the sweep had just changed. That column is now `center` and the audit
     re-runs under it. **The remaining non-ASCII in the repository is four
     justified cases**: unit strings that must match EC3's own text, French
     product names in a regex that must match real EPDs, and the author's name.

     **And the retired weighting vocabulary is out of figure LABELS, not just
     filenames.** Seven cells across three notebooks passed the raw PEWT key as a
     legend label, so "KDE, Variable" was printed where decision 199 settled on
     "market weights".

235. **2026-10-01, Stage 3. THE FIGURE NUMBERING, FULL STYLE COMPLIANCE AND THE
     CONFIDENCE INTERVALS ALL BELONG TO THE MANUSCRIPT WORK, NOT TO A
     REPOSITORY STAGE.** `[AUTHOR]` "Don't worry about figure renumbering yet.
     That'll depend on what we end up including in the manuscript." And, on the
     proposal Stage 3 wrote anyway: "to number figures, we need to decide what
     figures belong in the manuscript. To do that, we need to clearly understand
     the structure of the manuscript. You don't have enough information to do
     that right now ... This can't be done in isolation without reviewing the
     manuscript itself."

     This supersedes the Stage 3 prompt's instruction to propose a numbering and
     have it confirmed. The numbering cannot be settled before the figure
     SELECTION is, the selection needs the manuscript's structure, and no
     repository stage has it: about fifty images against the six or seven a
     Building and Environment paper carries. **Stage 4 does not do it either.**

     **TWO OF STAGE 3'S OPEN ITEMS HAVE THE SAME DEPENDENCY AND MOVE WITH IT.**
     Bringing the 31 figure cells that never call `figstyle.apply()` up to
     `FIGURE_STYLE.md`, and putting a confidence interval on every figure
     aggregate that needs one, are both PER-FIGURE work. Doing either for forty
     figures when six reach the paper is the wrong order, and which aggregates
     need an interval is a question about what the paper claims rather than
     about what the repository holds. All three are one pass, taken once the
     selection exists.

     **The cost of deferring is near zero**, which is what makes it the right
     call rather than a postponement. `figstyle.savefig` takes a STEM rather
     than a path, so applying any numbering later is a one-line change per
     figure cell; the duplicate-name guard makes a collision impossible; and
     `tests/test_figure_manifest.py` fails on any file a rename leaves behind
     without a generator.

     The sketch is in report deleted by decision 251 section 5 and the full file
     list with every generator is
     `outputs/tables/audits/TABLE_FigureManifest.csv`, so nothing has to be
     worked out twice.



236. **2026-10-01, Stage 3. THE BACKWARDS CONTROL RAN FOR THE FIRST TIME AND IT
     LOSES, WHICH IS WHAT SAYS THE RULE'S DIRECTION IS NOT ARBITRARY.**
     `[DELEGATED, 3 measured]` Stage 2j built `MixedBackwards` -- market weights
     BELOW the cutoff and uniform above, the rule the wrong way round -- as a
     control that should lose to both recommended rules, and left it unrun. The
     Stage 3 re-run of notebook 3 is the first time it has been scored.

     Pooled over the sixteen claims, as a percentage of the true level:

         known-share rule                      0.2025
         KDE, market weights                   0.2293
         FEASIBLE rule                         0.2325
         Lognormal, market weights             0.2365
         KDE, uniform weights                  0.2392
         Lognormal, uniform weights            0.2396
         BACKWARDS control                     0.2591
         Normal, uniform / market              0.3144 / 0.3165

     **It is worse than every fixed method except the two normals.** Putting
     market weights on the SMALL datasets, where decision 222 shows they make
     the estimate of the mean worse, and withholding them from the large ones,
     where they help, costs more than any fixed choice a reader could have made.

     **And adding it changed nothing else**: all 61 policies that already
     existed move by at most 1.1e-16 in pooled error, which is the control on
     the control.

         python -c "import pandas as pd; d=pd.read_csv('outputs/tables/TABLE_MixedPolicyRanking.csv'); print(d.nsmallest(9,'pooled_error')[['display','pooled_error']].to_string(index=False))"

237. **2026-10-02, Stage 3 review. THE SEVEN-POLICY SCORECARD COMPARES COLUMNS
     MEASURED IN TWO DIFFERENT MONTE CARLO EXPERIMENTS, AND ITS PUBLISHED COUNT
     IS 11 OF 16 RATHER THAN 10 WHEN THE COMPARISON IS MADE ON ONE. This
     NARROWS decision 206.** `[AUTHOR]` "Why are there two separate monte carlo
     experiments? That doesn't make any sense ... We should definitely rebuild
     the seventh column from the same 10,000 runs as the other six."

     **WHAT IS WRONG.** `TABLE_ClaimScorecardWithRule.csv` is built by
     concatenating the six-method scorecard, which comes from the study's main
     run against the true parents, with the feasible rule's row, which comes
     from the separate pass Stage 2j added at the end of notebook 3. The two
     passes spawn their own random streams, and on the design comparison they
     also resample their 2,500 design pairs separately.

     **MEASURED, the same six methods on the same sixteen claims, one pass
     against the other:** median difference 0.007 points of true level, and
     **0.70 on the design comparison**. The rule's smallest winning margin is
     0.10 and its margin on the design comparison is 0.03, so that claim is
     decided by which pass it was measured in. Taking all seven columns from
     the mixed pass, which already holds every one of them:

         across two passes (the figure today)   10 of 16
         within one pass                        11 of 16
         best of all seven, either way           2 of 16

     **DECISION 206'S REASON WAS SOUND AND ITS CONSEQUENCE WAS NOT.** A seventh
     policy changes `best_method`, `stakes`, `excess` and every win-share
     denominator, because those are properties of the SET compared, and a
     separate pass protected the six-method tables from that. But
     `metricset.rescore` already derives those columns separately for the two
     tables, so the protection did not need a separate EXPERIMENT.

     **THE FIX IS SAFE AND ITS CONTROL IS EXACT.** `plca.truth_run` and
     `plca.swap_run` each draw the uniform block ONCE per group or per pair,
     BEFORE the loop over methods, so adding a seventh entry to `methods`
     consumes no random numbers. The six existing columns must come back
     BIT-IDENTICAL; if they do not, something else moved and the run stops.
     Stage 4 owns it, as its first code change rather than after a run.

         python -c "
         import pandas as pd
         m=pd.read_csv('outputs/tables/TABLE_MixedPolicyScorecard.csv')
         CH=['KDE, Uniform','Lognormal, Uniform','Normal, Uniform','Feasible@80']
         p=m[m.method.isin(CH)].pivot(index='claim',columns='method',values='total_error')*100
         print(int((p.drop(columns=['Feasible@80']).min(axis=1)-p['Feasible@80']>0).sum()),'of',len(p))"

238. **2026-10-02, Stage 3 review. THE SCORECARD COUNTS ONE CLAIM TWICE AND THE
     DUPLICATE IS DROPPED. The denominator is 15, not 16.** `[AUTHOR]` "Let's
     get rid of 'what using 25% less of a material saves'."

     Decision 186 established that "what using 25 percent less of a material
     saves" is an exact algebraic function of "a material's share of the total"
     -- removing a quarter of a material's contribution makes the error in the
     first exactly 0.25 times the error in the second, so their RELATIVE errors
     are equal. Verified again on the seven-policy table: identical in all
     seven columns to **7.2e-16**. The Stage 3 prompt asked for the row to be
     dropped from the figure and it was not, so the figure presents one claim
     as two pieces of evidence, and both copies are losses for the rule.

     **WHAT MOVES WHEN IT GOES.** The rule is best of the four a reader can
     choose on **10 of 15** rather than 10 of 16, and the pooled figures rise
     because the dropped claim is one of the easiest: the rule 23.97 against
     24.66 for a kernel estimate with uniform weights and 24.73 for a
     three-parameter lognormal, where the sixteen-claim version reads 23.25,
     23.92 and 23.96. **These are the sixteen-claim numbers on two passes; they
     are superseded again by decision 237's rebuild and must be recomputed
     once, after it.** The caption keeps one sentence saying why the row is
     gone, because the identity is itself worth knowing: a quantity reduction
     is a deterministic fraction of a material's own share, so its accuracy IS
     that share's accuracy and no distributional assumption enters.

         python -c "
         import pandas as pd
         d=pd.read_csv('outputs/tables/TABLE_ClaimScorecardWithRule.csv')
         a=d[d.claim=='a material: its share of the total'].set_index('method').total_error
         b=d[d.claim=='using 25 pct less: its mean saving'].set_index('method').total_error
         print('max difference across all seven policies:', (a-b).abs().max())"

239. **2026-10-02, Stage 3 review. STAGE 4'S FIRST TASK IS THE CODE CHANGE, NOT
     THE RUN, AND THE ORDER IS REVERSED FROM WHAT THE STAGE 3 REPORT SAYS.**
     `[AUTHOR]` "Sounds good, let's change the order."

     The Stage 3 report's section 9 tells Stage 4 to re-run notebook 3 first,
     for two measurements Stage 2j had left pending. Sections 3b and 7 of the
     same report say both already ran in Stage 3: the backwards control scored
     0.2591 pooled, and the group-composition split "has happened and it is
     unchanged, because widening it needs a code change nobody made". So the
     instruction sends the next window to spend about 110 minutes regenerating
     what exists, after which the one genuinely missing column is still
     missing.

     **THE ORDER IS: the four code changes of the Stage 4 section -- the
     seventh column on one pass (237), the dropped claim (238), the feasible
     rule's composition column, and the compute/plot split for notebook 3's
     nine unrenderable figure cells -- then ONE notebook-3 run, then the
     bit-identity controls.** This is the "wrong first task for the next stage"
     failure the Stage 2j review already caught once, and it is recorded here
     because it has now happened twice.

240. **2026-10-02, Stage 3 review. ONE SET OF RUNTIMES, IN ALL THREE PLACES
     THAT CARRIED THREE DIFFERENT SETS.** `[AUTHOR]` "Let's update the
     estimated times. It's hard to tell because I'm running code in other
     repositories so sometimes it takes longer ... let's update it with our best
     guess."

     Stage 3 measured 36, 13, 110 and about 20 minutes for notebooks 1 to 4 and
     recorded that in its report, while in the same stage ADDING to the README
     a line reading "notebook 1 about 30 minutes, notebook 2 about 35, notebook
     3 about three and a quarter hours, notebook 4 about 20" -- so the deposit's
     own README overstated notebook 2 by a factor of 2.7 and notebook 3 by 1.8.
     `CONTEXT.md` carried a third set again, 30 / 35 / 45 / 15.

     **The published figures are now 35, 15, 110 and 20 minutes**, quoted to
     the nearest five minutes as wall clock with nothing else competing, with
     one sentence saying a busy machine can take half again as long. README,
     `CONTEXT.md` and the prompt file agree. The Stage 2f cell-by-cell profile
     stays in `CONTEXT.md` because it says WHERE the time goes; its totals are
     explicitly superseded.

241. **2026-10-02, Stage 3 review. THE pLCA SCATTER FIGURE GETS ITS (a) (b) (c)
     (d) PANEL LABELS, AND IT IS NOT URGENT.** `[AUTHOR]` "Sure, add in some
     figure panel labels. This won't be super relevant until we pick which
     figures are actually going into the manuscript though."

     The Stage 3 prompt asked for two changes to that figure: drop the
     highlighted single-pLCA points, and label the panels explicitly. The first
     was done, the second was not, and the Stage 3 report mentions only the
     first, so it reads as complete. No figure cell in any notebook labels its
     panels. Stage 4 does it cheaply; the full per-figure pass waits on the
     manuscript's figure selection with decision 235's other two items.

242. **2026-10-02, Stage 3 review. THE BANDWIDTH IS NOT UP FOR DEBATE AND THE
     REVIEW IMPLIED IT WAS. `silverman_guarded` STANDS, AND IT WAS NEVER THE
     BEST ON W1 OR ON THE pLCA -- IT IS INSURANCE ON THE WORST CASE.**
     `[AUTHOR ASKED]` "I thought we found that guarded Silverman outperformed
     both Scott and Silverman? Why is this still up for debate?"

     It is not up for debate and nothing in Stage 3 reopened it. What the Stage
     3 review actually found was a wrong SENTENCE in the report, not a wrong
     rule.

     **NO LATER STAGE RE-MEASURES THIS.** The author has now had to ask twice
     why it was being discussed, and the reason both times was a review finding
     about prose that read as a finding about the rule. `BW_METHOD =
     'silverman_guarded'` with `SILVERMAN_MIN_NEFF = 20` is settled by
     decisions 54 and 80, re-run as a stale audit in decision 233, and
     confirmed downstream here. The author's own statement of the reason is the
     one to carry into the paper: **the guard exists to avoid the small-n
     interquartile-range trap**, where at three to ten declarations the
     quartiles are interpolated between two order statistics and the robust
     scale estimate collapses. A later window that finds a sentence about
     bandwidths wrong should correct the sentence and say so, and must not
     present it as an open question. The sentence said Scott is "now best on two outputs" and printed two
     numbers that show it losing one of them; and it confined the exception to
     uniform weights when it occurs under both.

     **WHAT THE RULE WAS CHOSEN ON, because the record is spread over four
     decisions and is easy to misremember.** Decisions 54 and 80 chose the
     guard on the FIFTH PERCENTILE of held-out log-likelihood -- the small-n
     failure where a collapsing bandwidth does its damage -- and deliberately
     NOT on W1, because W1 falls monotonically as the bandwidth shrinks and
     would have chosen a spike. Decision 71 then recorded that pure Silverman
     beats the guarded rule against the known parent on 90.1 and 83.2 percent
     of datasets, and decision 75 recorded that the W1 optimum is at a
     threshold of 5 rather than 20. **So "guarded beats both" was never the
     finding.** The finding is that both Silverman forms beat Scott decisively,
     and the guard costs a fraction of a percent against pure Silverman in
     exchange for repairing the worst case at three to ten declarations.

     **ON THE CURRENT CORPUS, mean relative error over the five pLCA outputs:**

         uniform weights   Scott 29.23   Silverman 28.24   guarded 28.51
         market weights    Scott 30.06   Silverman 28.89   guarded 28.93

     Scott is worst on the mean under both weightings, which is why the rule
     stands. Per output Scott is best on ONE of the five, the uncertainty
     index, under BOTH weightings, and non-worst on a second; decision 195's
     "Scott is worst on every one of the five outputs" is withdrawn and
     decision 233's restatement of it is corrected here. **The guard now costs
     0.28 of a point under uniform weights and 0.04 under market weights**,
     where decision 195 recorded it BUYING 0.15 under market weights. The
     control fires: the four parametric methods move 0.0000 across all three
     rules.

         python -c "
         import pandas as pd
         d=pd.read_csv('outputs/tables/audits/TABLE_BandwidthDownstreamSummary.csv')
         k=d[d.method.str.startswith('KDE')]
         print((k.pivot_table(index=['method','output'],columns='bw_rule',values='rel_error')*100).round(3))"

243. **2026-10-02, Stage 3 review. THE RUNAWAY-TAIL FIGURE IS REPORTED AS AN
     AGGREGATE OVER ALL FITS, NOT AS THE SINGLE WORST ONE.** `[AUTHOR]` "Why do
     we care about the worst runaway fit? We shouldn't isolate any single fit.
     That's not what this study is about. We're looking at aggregated results
     across 10,000 datasets."

     Correct, and the review's own finding on it was the wrong shape: it
     argued about which single fit the maximum belongs to. The maximum exists
     in this project as a GUARD -- decision 149 is why level metrics are safe
     to report at all, and decision 152 watches `model_sd_ratio` as the
     sentinel -- and a guard is not a result.

     **SO THE ONE-LINE TRUNCATION REMEDY OF DECISION 199 IS STATED IN AGGREGATE
     TERMS.** On the current corpus, over 3,882 fits with no cap: the median
     fitted model's spread is **0.98 times the data's own**, **1.75 percent**
     exceed twice it, **0.39 percent** exceed three times, and **none** exceeds
     five times. A cap at twice the largest observed value takes the share above
     twice the data's spread from 1.75 percent to **0.03**, and against the
     known truth it costs nothing: four of the six methods do not move at all,
     one gains 0.00011 and one loses 0.00124. **That is the sentence the paper
     carries**, and it says the same thing as decisions 182 and 199 without
     resting on one fit. The single worst fit stays in the audit table as the
     sentinel it is.

         python -c "
         import pandas as pd
         d=pd.read_csv('outputs/tables/audits/TABLE_UpperTruncationSweep.csv.gz')
         u=d[d.cap=='none']
         print('fits', len(u), 'median ratio', round(u.model_sd_ratio.median(),3))
         for t in (2,3,5): print(f'above {t}x: {100*(u.model_sd_ratio>t).mean():.3f} pct')"

244. **2026-10-02, Stage 3 review. THE FIGURES NEEDING A RESTYLE ARE 29, NOT
     31, AND DECISION 234'S "30 OF 36" IS THE SAME SLIP.** `[AUTHOR]` "Sure,
     change 31 to 29. This feels super nitpicky."

     It is, and it is recorded only because the number is handed forward as the
     size of a job. `TABLE_FigureStyleCompliance.csv` says 8 of 37 figure cells
     call `figstyle.apply()`, so **29** do not. The 31 is 37 minus the 6 cells
     that call all FOUR style helpers, which is a different count attributed to
     the wrong one. Decision 234 made the same substitution at the older total.

         python -c "
         import pandas as pd
         d=pd.read_csv('outputs/tables/audits/TABLE_FigureStyleCompliance.csv')
         print('do not call apply():', int((~d['apply']).sum()), 'of', len(d))
         print('call all four:', int(d[['apply','savefig','finish','overlaps']].all(axis=1).sum()))"

245. **2026-10-02, Stage 3 review. THE SCORECARD TITLE STAYS A BOX COUNT AND
     STAGE 4 UPDATES THE NUMBER IN IT WITHOUT ASKING AGAIN.** `[AUTHOR]`
     "Scorecard title looks great."

     Decision 230 records the author choosing that title from four options, so
     a changed count could otherwise look like a question to re-open. It is
     not. The FORM is approved -- the title counts the solid boxes, which is
     what the boxes say -- and the count moves with decisions 237 and 238: the
     rebuild on one truth pass and the dropped duplicate claim take it from
     "10 of 16" to **10 or 11 of 15**, forecast at 11.

     **The claim that decides 10 against 11 is the design comparison, where the
     rule and the best uniform-weighted method differ by 0.03 points of true
     level against a run-to-run spread of 0.70.** They are indistinguishable
     there. Stage 4 writes whichever count the rebuilt table gives and says in
     one sentence that the deciding claim is a tie, rather than defending the
     integer.

246. **2026-10-02, Stage 4. THE SEVENTH SCORECARD COLUMN IS SCORED ON THE MAIN
     TRUTH PASS, and the control that says it moved nothing is EXACT on the
     numerator and CANNOT be on the ratio. This IMPLEMENTS decision 237.**
     `[AUTHOR]`

     `Feasible@80` is added to `prob_models` before notebook 3's main truth run
     and main design swap and passed in `methods`, so the six-method scorecard
     and the seven-policy one are one Monte Carlo experiment rather than two.
     Both passes draw their uniform block ONCE per group and per pair, BEFORE
     the loop over methods, so a seventh entry consumes no randomness.

     **THE SIX-METHOD TABLES KEEP EXACTLY THE ROWS AND THE ORDER THEY HAD.**
     The rule's rows are split off into four tables of their own --
     `TABLE_PLCATruthRule.csv.gz`, `...BuildingRule`, `...InterventionRule` and
     `TABLE_PLCADesignSwapRule.csv.gz` -- so none of the thirteen cells that
     read the truth tables needed a filter and none of them changed.

     **WHAT THE CONTROL CAN AND CANNOT SAY, and the distinction is not
     pedantry.** `error` is the mean absolute error WITHIN one method, so no
     other method can touch it: it agrees to **0.000e+00** across all 90
     (claim, method) cells, and so does `error_portfolio`. `scale` is the mean
     TRUE level over every row of a claim, and the truth column repeats
     identically once per method, so with seven it is a mean over seven
     identical blocks instead of six -- the same number by a different
     summation order, differing in the last bit. It moves by **2.2e-16**, and
     `total_error`, their ratio, inherits that.

     **AND THE EXACT STATEMENT IS PINNED BY A TEST RATHER THAN BY THE RUN**,
     because the notebook's two paths differ in more than the seventh policy.
     `tests/test_metricset.py::test_adding_a_method_leaves_every_other_method_s_error_bit_identical`
     adds a seventh method with genuinely different values and asserts every
     other method's `abs_error`, `bias_raw`, `error` and `error_portfolio` are
     unchanged to the last bit, through both `recovery_table` and
     `per_unit_error`.

     **TWO FALSE ALARMS THIS CONTROL RAISED BEFORE IT WAS RIGHT, both worth
     recording because both look like the real thing.** First, recomputing the
     recovery table from the CSV the run had just written moved `abs_error` by
     2e-16: **`to_csv` followed by `read_csv` does not round-trip a float64 to
     the last bit in this pandas**, so the control was failing on the CSV and
     not on anything the seventh policy did. Second, comparing the in-memory
     seven-policy frame against the six-method scorecard READ BACK OFF DISK
     failed the same way, at 9e-17. Both sides are now in memory.

         python -m pytest tests/test_metricset.py -q

247. **2026-10-02, Stage 4. THE SCORECARD IS FIFTEEN CLAIMS. This IMPLEMENTS
     decision 238.** `[AUTHOR]` "Let's get rid of 'what using 25% less of a
     material saves'."

     It is exactly 0.25 times "a material: its share of the total" -- a
     quantity reduction removes a deterministic fraction of a material's own
     contribution, so no distribution enters and the two RELATIVE errors are
     equal, verified at 7e-16 across every policy (decision 186). Carrying both
     made every count have a denominator of 16 where it should have been 15.

     `metricset.DUPLICATE_CLAIMS` records what was removed, with the factor and
     the claim it duplicates, so a later stage adding a claim can check it is
     not a third copy. The underlying column is untouched in
     `TABLE_PLCATruthIntervention.csv.gz`.

     **EVERY POOLED FIGURE AND EVERY "N OF 16" IN THE DECISION LOG MOVES WITH
     THE DENOMINATOR, and no measurement does.** Decision 217's 9 of 16 and 16
     of 16, decision 230's 10 of 16 and 2 of 16, and the pooled errors of
     decisions 213, 217, 218, 221 and 236 are all over the sixteen-claim set.
     **A manuscript session must take these from the rebuilt tables and not
     from the decision log.**

     **AND THE SENTENCE THE PAPER KEEPS** is the identity itself: a quantity
     reduction is a deterministic fraction of a material's own share, so its
     accuracy IS that share's accuracy and no distributional assumption enters.
     That is a finding, and it is why the row could go.

248. **2026-10-02, Stage 4. ALL 37 OF 37 FIGURE CELLS NOW REDRAW ON THEIR OWN,
     AND ONE OF THEM WAS DRAWING FROM THE NOTEBOOK'S RANDOM STREAM.**
     `[AUTHOR]` This finishes the compute/plot split Stage 3 did for notebooks
     1 and 2; notebook 3 had nine of thirteen figure cells that could not be
     rendered without the whole run.

     **THE SERIOUS ONE IS THE RANDOMNESS.** `FIG_PLCAVisualizeUQFits` drew its
     whole-building totals with `rvs(..., random_state=rng)` INSIDE the figure
     cell -- 72 blocks of 10,000 draws from the notebook's own Generator -- so
     redrawing that figure moved every number after it. The standing constraint
     is that all randomness comes from an explicitly passed Generator, and it
     did; what nobody had noticed is that a FIGURE was consuming it. The draws
     move to a compute cell in the same place in the stream, which preserves it
     exactly: same nesting order, same count.

     The other three changes are mechanical. The display constants and the two
     comparison helpers move into the setup cell -- the renderer executes every
     cell up to and including the one defining `OUT`, then one figure cell --
     and the helpers take their frames as ARGUMENTS rather than reading
     globals. Five frames that lived only in kernel memory are persisted:
     `TABLE_PLCAInterMethodDistance.parquet` and four `TABLE_PLCAExample*`
     tables carrying the three case-study pLCAs' values, fitted densities, W1
     matrices and whole-building draws.

     **A HELPER DEFINED ABOVE A CELL MUST NOT EVEN MENTION A NAME THAT CELL
     ASSIGNS.** `compare_results` took a parameter called `df_plca`, which is
     also the global the pLCA cell creates far below it, and
     `tests/test_notebooks.py`'s forward-read guard cannot tell a parameter
     from the mistake it exists to catch. Renamed to `plca_results`.

     **AND AN EXPLORATORY CELL IS REMOVED.** It recomputed the case-study
     variance over all 2,500 groups -- which the cell below it computes again
     -- and drew the highlighted single-pLCA points the author asked to drop in
     Stage 3, coloring the four materials of the group with the largest spread,
     which is a maximum picked out of 2,500 and so the least representative
     group there is. Nothing below it read any name it defined.

     **WHAT IT BUYS.** A figure round in notebook 3 was the full run. On the
     heaviest of its figures, a scatter of about 1.8 million points, it is
     **27 seconds** with a smoke run competing for the processor.

         python audits/figure_manifest.py
         python -m pytest tests/test_figure_manifest.py -q

249. **2026-10-02, Stage 4. DECISION 241 IS WITHDRAWN: the panel labels it asks
     for have been on that figure since 2026-05-12. The defect next to them is
     real and is fixed instead.** `[DELEGATED, 4 checked]`

     Decision 241 records that the Stage 3 prompt asked for (a) (b) (c) (d)
     panel labels on the pLCA scatter figure alongside dropping the highlighted
     single-pLCA points, that the points were dropped and the labels were not,
     and that the Stage 3 report did not say so. The labels are there, in the
     cell and in the committed PNG, and they predate Stage 3 by five months.

         git log -S "alphabet[ires]" --oneline -- notebooks/03_CompareUQ_PerformPLCA.ipynb

     **What IS wrong with that figure is visible in the same image**: the top
     row's x axis labels print on top of the bottom row's two-line titles. The
     row spacing is fixed.

     **The lesson is the one this project keeps relearning.** The review formed
     the finding from the prompt's instruction rather than from the figure, so
     it reported a defect that had been fixed and missed one that was next to
     it. `reports/START_HERE.md` already requires a reproduce-command for every
     review finding; an image is reproduced by looking at it.

250. **2026-10-02, Stage 4. THE DEPOSIT SAYS WHICH CORPUS THE PAPER DESCRIBES,
     AND THE README'S ACCOUNT OF THE WEIGHTS WAS WRONG IN TWO WAYS.**
     `[AUTHOR]`

     **The corpus.** A reader arriving from the paper's citation saw 49
     `corpus_*` directories and nothing saying which one the study used.
     `data/processed/README.md` says it: **`corpus_2026-09-25`**, named by
     `CORPUS.json`. Every other directory is superseded and is kept because a
     corpus here is immutable -- a change to generation writes a new one beside
     the old so the two can be diffed file by file, which is how a change is
     proved to have moved only what it was meant to (decisions 58, 64, 125).
     They are not a second dataset and no number in the paper comes from any of
     them. `corpus_2026-09-15b`'s two derived replay caches, 6.6 MB of cache
     for a superseded corpus whose data the deposit does not carry, are
     untracked; `.gitignore` explicitly left that removal to this stage.

     **THE WEIGHTS. The README said both arms draw market shares from a flat
     Dirichlet. Neither arm does, and the error inverts what the comparison
     means.** Since decision 190 the empirical arm uses
     `weighting.coherent_weights` at a coherence of 0.5, and on the synthetic
     arm the share attached to a product group is that group's TRUE share in
     the parent, exact to 1.1e-16 (decision 212). So uniform against market on
     the synthetic arm is **ignoring a known share against using it**, never
     guessing against knowing -- which is precisely the objection the author
     raised in Stage 2j and which decision 212 answered. A reader checking the
     repository against the paper would have drawn the wrong conclusion from
     the README alone.

     **The figure manifest is in the README itself** rather than referenced,
     because the audit table it comes from is under `outputs/tables/audits/`
     and is gitignored, so the deposit would not carry it. 55 images grouped by
     notebook, each with what it shows. **No file carries a figure number**
     (decision 235), and the `FIG_` and `SUPP_` prefixes record what the
     generating cell declares itself to be rather than where the manuscript
     puts it. `WassVsResultDiff` was the one image with neither prefix and is
     written as `FIG_WassVsResultDiff`; the old pair is in `archive/`.

     **The spelling and vocabulary sweeps are finished**, which closes the item
     Stage 2g handed to the deposit tidy-up. Fourteen files, all comments and
     prose. `characterisation` survives in exactly one place, the published
     title of Marsh, Lewis, Hattam and Allen (in press), and the four files
     with non-ASCII characters are the justified ones: the author's name,
     French product names a regex must match, and unit strings EC3 itself
     writes. The retired weighting vocabulary is out of the COLUMN NAMES --
     `variable`, `variable_wins`, `rank_vs_variable_target` and
     `variable_closer_pct` all become `market*` -- and out of one figure the
     Stage 3 sweep missed, whose title asked "Does variable weighting help?".
     **The stored `method` values keep "Variable"** by decision 199, so a
     pivoted table still shows `KDE, Variable` as a column header: that is the
     join key, not a label.

251. **2026-10-02, Stage 4. THE SUPERSEDED REPORTS ARE DELETED, and nothing
     outstanding lived only in them.** `[AUTHOR]` CLAUDE.md's retention rule
     keeps only the CURRENT stage's report; this is the last stage, so six
     `HANDOFF_stage-*.md` files, `STAGE_REPORT_2j.md` and `STAGE_REPORT_3.md`
     go. Git history retains them, which is what makes it safe, and the
     repository is published alongside the paper where a reader has no use for
     the editing process that produced it (decision 59).

     **The check was done per item rather than per file.** The Stage 2h handoff
     carries the cumulative open list and every item on it is either resolved
     in this decision log or is this stage's own work; the two it assigned to
     "the deposit tidy-up" are the British spellings and the retired
     vocabulary, both in decision 250. Stage 2j's open items were settled by
     decision 207 and Stage 3's by decision 235 and the Stage 4 prompt. The one
     manuscript item that lives in neither -- restating the coverage claim from
     the rebuilt table -- is entry 34 of
     `reports/MANUSCRIPT_discrepancies.md`, which survives.

     **`reports/STAGE_PROMPTS.md` IS NOW A RECORD IN FULL** and no section of
     it may be edited. The only part still maintained is the configuration
     block at the top, which supersedes any value quoted inside a sent stage
     and has to stay true for as long as anyone reads the code; this stage
     added its two production changes to it.

252. **2026-10-05, MANUSCRIPT. THE ELEVEN STALE PLACES ARE FIXED, AND THE
     PAPER'S HEADLINE NUMBERS ARE RESTATED HERE FROM THE TABLES.** `[AUTHOR]`
     "If there are eleven places the record is now stale, let's go fix them!
     This repository shouldn't be stale. We're getting ready to write the
     manuscript so we want this to be as up to date and tidy as possible."

     Found by reading the tables against the decision log while planning the
     manuscript, and recorded in `reports/MANUSCRIPT_NARRATIVE.md` section 5.
     **The log was the stale side in every one.** Everything below is on
     `corpus_2026-09-25` at `weight_rho = 0.5`.

     **THE SAFE-LEAD RATIO IS 2.28 AGAINST 2.33, NOT 2.13.** At the study's own
     four materials, the chance that the choice of method changes which
     material leads crosses 1 percent at a top-two contribution ratio of 2.28
     [2.18, 2.39] parametric and 2.33 monotone; 5 percent at 1.7 against 1.6;
     10 percent at 1.53 against 1.51. Pooled over group sizes the 1 percent
     crossing is 2.5 against 2.3. **The anchor is unchanged**: Marsh, Lewis,
     Hattam and Allen (in press) put their Concrete-Precast staircase at a
     top-two ratio of 1.02. Decisions 107 and 144 carry superseding notes.

         python -c "import pandas as pd; d=pd.read_csv('outputs/tables/TABLE_PLCARatioCrossings.csv'); print(d[d.nmats=='4'][['level','ratio','ratio_lo','ratio_hi','ratio_isotonic','prose_text']].to_string(index=False))"

     **THE SCORECARD IS FIFTEEN CLAIMS AND EVERY POOLED FIGURE IN THE LOG IS
     OVER SIXTEEN.** Decision 247 flagged this generically; these are the
     values. Pooled over the fifteen, as a percentage of each claim's own true
     level: the size rule **23.96**, a kernel estimate with uniform weights
     24.66, a three-parameter lognormal with uniform weights 24.73, a normal
     32.50, and a kernel estimate with market weights 23.58. The rule is the
     best of the four a reader can choose on **10 of 15** and the best of all
     seven on 2 of 15. **The value of market-share data is 23.97 to 20.89, which
     is 3.08 points or 12.8 percent**, where entry 183's sixteen-claim version
     reads 23.24 to 20.25 and 2.98 points. The relative figure did not move.

     **THE FIVE STATEMENTS, REBUILT.** The design comparison: at a claimed 5
     percent saving the truth is **0.599**, the six methods span **0.609 to
     0.622**, and every method is within **0.023** of the truth. The building
     total: every method understates the 90th percentile, by **0.125 to 0.567**
     on a total averaging 4.24, and at a budget the truth meets 90.0 percent of
     the time the six report **86.9 to 92.0**. The specification cap: against a
     true mean saving of **6.48 percent** of the building the six report 5.65 to
     7.27, and asked for the chance of achieving at least 5 percent the truth is
     **24.7** and the six span **23.9 to 33.7**. Decision 144's own numbers are
     superseded wholesale rather than patched, per decision 201.

     **THE TWO-PARAMETER LOGNORMAL COMPARISON IS RE-RUN AND IT GOT STRONGER.**
     `audits/lognormal_variants.py` was last run on the superseded corpus and
     carried no provenance stamp; it now stamps the corpus and the weight rule.
     On 1,500 shipped-corpus datasets scored against the KNOWN PARENT under
     uniform weights, median relative gain over the two-parameter lognormal:

         band       3-par lognormal   gamma   normal   kernel estimate
         3-9             +0.4          +0.3    +2.3        -0.0
         10-99           -4.1          -0.4   +16.7        -3.0
         100-999        -22.9         -14.5   +23.1       -33.2
         1000+          -25.0         -17.6   +36.3       -48.1

     The kernel estimate is **33 to 48 percent** closer to the truth than a
     two-parameter lognormal above 100 declarations, where decision 167 reports
     31 to 41, and beats it on **67.4 percent** of datasets against that entry's
     71.5. The three-parameter lognormal is **23 to 25 percent** closer above
     100 declarations and beats it on 62.7 percent. Closest family overall:
     kernel estimate 41.5 percent, three-parameter lognormal 19.6,
     two-parameter 16.9, gamma 12.5, normal 9.5. **Decision 167's ORDERING
     stands and its levels are superseded.**

     **AND THE REASON HAS A ONE-LINE PROOF, WHICH THE PAPER SHOULD CARRY.** The
     author asked whether the two-parameter lognormal also imposes a shape. It
     does, exactly: for a two-parameter lognormal, **skewness = CV^3 + 3 CV**.
     Once the spread is matched the skewness is decided, so it has no free
     shape parameter at all, which is the same disability as the normal's fixed
     skewness of zero and not a milder version of it. The THREE-parameter fit
     escapes because **skewness is shift-invariant**: sigma alone sets the
     skewness and the threshold then sets the coefficient of variation
     independently, so the two can be matched together. A kernel estimate
     constrains neither. **Measured on the real categories with ten or more
     declarations: only 21.3 percent have a skewness within 25 percent of what a
     two-parameter lognormal of their own spread must have**, the median
     category is 0.58 times as skewed as the curve requires, and 23.6 percent
     are more skewed. So the ladder normal, two-parameter lognormal,
     three-parameter lognormal, kernel estimate is a ladder of SHAPE FREEDOM,
     and it predicts the study's own ordering before any fit is run.

     **DECISION 65'S PROHIBITION HAS LOST ITS PREMISE AND IS AN OPEN
     QUESTION.** It forbids comparing a cross-validated score across weighting
     schemes, on the grounds that the empirical weights are an exchangeable flat
     Dirichlet draw so the uniform-weighted fit is the better predictor by
     construction. Since decision 190 those weights are coherent blocks at
     rho = 0.5 and are correlated with the values, so a random half preserves
     the correlation and the exchangeability argument does not apply. On the
     current table the market-weighted lognormal beats its uniform twin (0.3373
     against 0.3462) while the market-weighted kernel estimate loses (0.3907
     against 0.3764), which is not what the prohibition predicts. **Nothing is
     reversed here and no number moves**; the question of whether the empirical
     arm can now say anything about weighting is put to the author.

     **THE EMPIRICAL ARM'S HEADLINE HAS MOVED AND NOW CONTRADICTS THE DRAFT.**
     Cross-validated over the 127 categories that reach n = 10, the closest
     method is a three-parameter lognormal on **53.6 percent** of categories
     (26.8 market, 26.8 uniform) against the kernel estimate's **26.0** and the
     normal's 20.5. The manuscript says the empirical arm corroborates that the
     kernel estimate is best. It does not. Decision 66 is why -- the criterion
     and the size mix, not a corpus-versus-arm disagreement -- and the paper has
     to say it rather than assert agreement.

     **SEVEN ORPHAN TABLES ARE DELETED AND `CONTEXT.md`'s INVENTORY IS
     CORRECTED.** `TABLE_ReductionWinner`, `TABLE_ReductionFitVersusAnswer`,
     `TABLE_ReductionIncremental`, `TABLE_ReductionMarginalVersusPartial`,
     `TABLE_ReductionPartialDependence`, `TABLE_ReductionChoiceIncrements` and
     `TABLE_SizeVersusMaterial` sat in the top level of `outputs/tables/`, were
     dated 2026-09-17 to 2026-09-21, and **had no producer anywhere in the
     repository** -- `CONTEXT.md` attributed five of them to notebooks that do
     not write them. That is the exact condition decision 56 exists to prevent
     and its precedent is deletion. Five are duplicated as `AUDIT_*` in
     `outputs/tables/audits/`, where `audits/metric_reduction.py` does write
     them and where decision 56 says audit output belongs; the inventory now
     names that script as the producer. `TABLE_SizeVersusMaterial` held the
     empirical size crossover that decisions 136 and 183 both WITHDREW as
     unmeasurable on 127 categories, so it carried retired numbers as well.
     **No table in `outputs/tables/` now predates the regeneration.**

     **TWO SMALLER CORRECTIONS.** `FIG_PLCATruth` printed "KDE, Variable" and
     "Lognormal, Variable" in its row labels, which decision 199 retired from
     every axis label and panel title; its cell now maps through
     `fitting.display_method` and the figure is redrawn. And
     `TABLE_FiveStatements.csv` describes the quantity reduction as
     "method-independent" with no qualifier, which is the PORTFOLIO reading that
     decision 174 corrected -- per building every method is 10.0 to 13.4 percent
     out. **The cell's note is fixed and the table on disk still carries the old
     wording until notebook 3 is next run**, which is a prose column rather than
     a number, so nothing is re-run for it.

     **WHAT IS NOT FIXED, AND WHY IT IS NOT A DEFECT.** The "CURRENT CANONICAL
     NUMBERS" block in `reports/MANUSCRIPT_discrepancies.md` is dated
     2026-09-17 and says it "beats anything below it". It no longer does, and it
     is a dated record of what was true then rather than a live reference, so it
     gains a header saying so rather than being rewritten.
