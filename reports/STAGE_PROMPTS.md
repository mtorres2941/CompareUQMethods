# Claude Code prompts: rework of the "Comparing UQ Methods" analysis

## Status, and what may be edited

**Stages 0 through 2h have been sent and run. Their text below is a RECORD of
what those sessions were given, and it is not to be edited.** Correcting a stage
after it has run makes the file disagree with what actually happened, which
destroys the only account of why a session did what it did. Where a sent stage
describes a parameter or a count that has since changed, the sent text stands and
the block below supersedes it.

**Stage 2j has now been sent and run as well, 2026-09-25, and its text below is
a RECORD on the same terms.** Only Stage 3 and Stage 4 are live and may be
edited. Where the Stage 2j text says "handoff" it means what is now called the
stage report, `reports/STAGE_REPORT_2j.md`; the specification changed between
that section being written and being run, and the section is not edited for it.

**Stage 2h changed the production path in three places and every number below
this line was re-read after it.** The real categories now draw their market-share
weights by a different rule; the synthetic corpus was regenerated; and the code
that draws from a known distribution in order to check answers against it was
repaired. Anything in Stages 0 through 2g that quotes a synthetic number, or a
weighted number from the real arm, describes a state of the project that no
longer exists.

## Current configuration, which supersedes any value stated in a sent stage

**Read from the code on 2026-09-25, at the close of Stage 2h and AFTER the
regeneration.** Every value below was printed by a script that ran to completion
and whose output was checked for a traceback before it was used. If any of these
disagrees with a number in a stage section above, this block is right.

The previous version of this block was read at the close of Stage 2g, was stale
on the generator, and was missing the empirical weight rule and everything in
`fitting`, `families`, `customstats` and `flip` -- because the script that
produced it raised `KeyError: 'count'` partway through and the output was pasted
without being read.

GENERATOR -- genconfig.DEFAULT (src/genconfig.py)
        genconfig.comp_exkurt_hi                   60.0
        genconfig.comp_exkurt_lo                   -1.2
        genconfig.comp_sd_log10_hi                 0.3
        genconfig.comp_sd_log10_lo                 -0.7
        genconfig.comp_skew_hi                     8.0
        genconfig.comp_skew_lo                     -3.0
        genconfig.cv_log10_hi                      1.2041199826559248
        genconfig.cv_log10_lo                      -2.3979400086720375
        genconfig.cv_log10_mean                    0.329
        genconfig.cv_log10_sd                      0.7838
        genconfig.k_max                            5
        genconfig.k_min                            1
        genconfig.market_share_alpha               1.0
        genconfig.max_component_retries            12
        genconfig.max_low_tail_truncated           0.15
        genconfig.max_parent_mean_over_median      25.0
        genconfig.max_parent_retries               20
        genconfig.min_mode_sd_frac                 0.15
        genconfig.min_q1_over_iqr                  0.2
        genconfig.mode_coupling                    1.0
        genconfig.mode_share_alpha                 10.0
        genconfig.overlap_log10_hi                 0.146128035678238
        genconfig.overlap_log10_lo                 -0.5228787452803376
        genconfig.overlap_statistic                min_adjacent
        genconfig.point_weight_alpha               1.0
        genconfig.position_skew                    5.0
        genconfig.seed                             42
        genconfig.separation_dispersion_frac       0.0
        genconfig.shoulder_body                    narrow
        genconfig.shoulder_frac                    0.0
        genconfig.trunc_iqr_mult                   3.0
        genconfig.trunc_rule                       log
        genconfig.strata[s1_3_9        ]        n 3 to 9, 2500 datasets
        genconfig.strata[s2_10_99      ]        n 10 to 99, 2500 datasets
        genconfig.strata[s3_100_999    ]        n 100 to 999, 2500 datasets
        genconfig.strata[s4_1000_9999  ]        n 1000 to 9999, 2500 datasets
        genconfig.probe[probe_10k_100k]         n 10000 to 100000, 50 datasets
    
    THE EMPIRICAL WEIGHT RULE -- THE KNOB REPORTED AT 0.5 IS `WEIGHT_RHO`
        empirical.WEIGHT_RHO                     0.5
        (the "coherence" of the handoff. empirical.prepare takes it as the
         keyword `rho` and passes it to weighting.coherent_weights)
        weighting.BLOCKS_MIN                     1
        weighting.BLOCKS_MAX                     5
        THE BLOCK COUNT RULE: weighting.draw_blocks draws k uniformly on
        [BLOCKS_MIN, BLOCKS_MAX] and INDEPENDENT of n, which is how the
        generator draws its own component count. empirical.prepare passes
        k=None so draw_blocks supplies it.
        weighting.coherent_weights defaults:
            k                          None
            rho                        1.0
            block_alpha                1.0
            point_alpha                1.0
            per_block                  8
            return_blocks              False
        empirical.prepare defaults:
            path                       /Users/martin.torres/Library/CloudStorage/Dropbox/Work/CUBoulder/Dissertation/Coding/CompareUQMethods/data/raw/ec3_raw_ecc_2026-08-14.csv.gz
            alpha                      1.0
            mult                       3.0
            min_n                      3
            split                      True
            ceiling                    True
            rho                        0.5
    
    EMPIRICAL ARM -- src/empirical.py
        empirical.DIRICHLET_ALPHA                  1.0
        empirical.CLEAN_IQR_MULT                   3.0
        empirical.MIN_N                            3
        empirical.MASS_ECC_CEILING                 100.0
        empirical.MAX_MASS_PER_UNIT                {'vol': 12000.0, 'area': 8000.0, 'length': 1000.0}
        empirical.MAX_ECC_PER_UNIT                 {'vol': 5000.0, 'area': 5000.0, 'length': 1000.0}
        empirical.MASS_UNIT_TYPE                   weight
        empirical.SPLIT                            True
        empirical.SOURCE                           /Users/martin.torres/Library/CloudStorage/Dropbox/Work/CUBoulder/Dissertation/Coding/CompareUQMethods/data/raw/ec3_raw_ecc_2026-08-14.csv.gz
    
    FITTING -- src/fitting.py
        fitting.FIT_METHOD                       mle
        fitting.BW_METHOD                        silverman_guarded
        fitting.W1_ROUTE                         trapezoid
        fitting.SCORE_GRID_POINTS                20000
        fitting.W1_TAIL_TERM                     True
        fitting.W1_TAIL_QUANTILE                 0.999999
        fitting.FAMILIES                         {'normal': (<function fit_normal_mle at 0x131c45f80>, <function make_normal at 0x131c462a0>), 'lognormal_2p': (<function fit_lognorm2_mle at 0x131c46020>, <function make_lognorm at 0x131c46340>), 'lognormal_3p': (<function fit_lognorm3_profile at 0x131c46160>, <function make_lognorm at 0x131c46340>), 'lognormal_offset': (None, <function make_lognorm at 0x131c46340>), 'gamma': (<function fit_gamma_mle at 0x131c46200>, <function make_gamma at 0x131c463e0>), 'weibull': (<function fit_weibull_mle at 0x131c46980>, <function make_weibull at 0x131c46a20>)}
        fitting.PEWT                             ['Normal, Uniform', 'Normal, Variable', 'Lognormal, Uniform', 'Lognormal, Variable', 'KDE, Uniform', 'KDE, Variable']
        fitting.WT_DISPLAY                       {'Uniform': 'uniform weights', 'Variable': 'market weights', 'Oracle': 'known market shares'}
        fitting.WT_DISPLAY_SHORT                 {'Uniform': 'uniform', 'Variable': 'market', 'Oracle': 'known'}
    
    FAMILIES -- src/families.py
        families.PROFILE_DELTA_LO_FRAC            0.25
        families.PROFILE_DELTA_HI_FRAC            1000.0
        families.PROFILE_GRID_POINTS              400
    
    CUSTOMSTATS -- src/customstats.py
        customstats.SILVERMAN_MIN_NEFF               20.0
    
    FLIP -- src/flip.py
        flip.FLIP_THRESHOLDS                  {0.01: 0.0029, 0.05: 0.015, 0.1: 0.032}
        flip.PROSE_MAX_SIGFIGS                4
        RECALIBRATED 2026-09-25 by author decision, at the close of Stage
            2h. These REPLACE 0.0018, 0.011 and 0.025, which were
            calibrated on the superseded corpus and had all three fallen
            outside their own recomputed 95 pct intervals. Notebook 1
            reads this constant and was re-run on it.
    
    RECOVERY -- src/recovery.py
        recovery.RECOVERY_GRID_POINTS             10000
        recovery.TAIL_QUANTILE                    0.999999999
        recovery.TAIL_GRID_POINTS                 2000
    
    PLCA -- src/plca.py
        plca.TRUTH_SCHEME                     market
        plca.TRUTH_SCHEME_SAMPLING            uniform
        plca.COMPARISON_MARGINS               (1.0, 1.05, 1.2)
    
    MIXED POLICY -- src/mixedpolicy.py, added by Stage 2j
        mixedpolicy.MIXED_THRESHOLD           80
        mixedpolicy.FEASIBLE_ABOVE            KDE, Uniform
        mixedpolicy.FEASIBLE_BELOW            Lognormal, Uniform
        mixedpolicy.LARGE_METHOD              KDE, Variable
        mixedpolicy.SMALL_METHOD              Lognormal, Uniform
        mixedpolicy.SWEEP_THRESHOLDS          every 10 from 10 to 200,
                                               plus 3, 300, 500, 1000,
                                               3000, 10000  (27 points)
        THE RULE THE PAPER RECOMMENDS is the FEASIBLE one: uniform
        weights throughout, the FAMILY switching at the cutoff. The
        pair above it -- KDE with market weights over a lognormal with
        uniform weights -- needs market shares nobody publishes and is
        a value of information, not a method. Decision 216.
        THE PUBLISHED CUTOFF RANGE is 40 to 170, the feasible rule's
        own indistinguishable band under a paired difference test
        against a FIXED reference. Do NOT publish the argmin's own
        bootstrap interval of [50, 140]: it answers where the curve's
        lowest point lands, not which cutoffs a reader can use.
        The CLAIM-level constant is 80; decision 142's 81 is a
        FIT-level argmin and stays that, and must not be printed as a
        claim-level number. Decisions 218, 220, 224.
        AND A SECOND, TIGHTER BAND ANSWERS A DIFFERENT QUESTION:
        knowing MARKET SHARE is significantly harmful below 80
        declarations and significantly helpful above 100, statistically
        zero in between. Decision 222.
    
    ACTIVE CORPUS
        data/processed/CORPUS.json               corpus_2026-09-25
    
    === DUMP COMPLETE, no exception ===

**`LOGFIT_OFFSET = 0.5` still exists in `fitting.py` but no production path
reads it**: it is used only by the superseded `fit_pewt_models`, by the named
comparison family `lognormal_offset`, and by two audit scripts.

**`flip.FLIP_THRESHOLDS` IS NO LONGER STALE AND IS NO LONGER WAITING ON THE
AUTHOR.** This paragraph said it was until 2026-09-25; the instruction was given
at the close of Stage 2h and taken. The constants are now the recomputed values
at two significant figures -- 0.0029 [0.00235, 0.00361], 0.015 [0.01325,
0.01716] and 0.032 [0.02868, 0.03505] -- and all three sit inside their own
intervals, where the superseded 0.0018, 0.011 and 0.025 sat outside them by
factors of 1.62, 1.37 and 1.26. They rose because the regenerated corpus is more
dispersed, so a given flip probability corresponds to a larger absolute model
distance: the same scale effect that raised every goodness-of-fit score without
any fit getting worse. Notebook 1 reads them to turn a per-dataset weighting
risk into a probability and has been re-run, which moved the mean probability
that unknown market shares change which material leads from 0.9870 to 0.9734 at
the 1 percent level, 0.8642 to 0.8144 at 5 percent and 0.7105 to 0.6524 at 10
percent. The control that says the change reached only what it should: the
separation columns, which do not read the constant, are bit-identical.

The empirical arm is 147 datasets and 116,766 values, drawn from 138 queried
categories; the frozen extract behind it holds 120,280 records, all of which the
database flags as product declarations rather than industry averages. The corpus
is 10,000 datasets plus a 50-dataset probe set.

The practitioner threshold is **81 declarations**, unchanged across the
regeneration, with the band of equally good choices now **68 to 106** where the
old corpus gave 68 to 97.

## How to use this file

Each step below is one paste into Claude Code. Everything else, including
git, is handled inside the prompts.

| # | Where | What to paste | Then |
|---|---|---|---|
| 1 | New window | Project Brief + Stage 0 | Read the handoff, bring it to chat |
| 2 | Same window | Stage 1 | Read the handoff, bring it to chat |
| 3 | New window | Stage 2a | Read the handoff, bring it to chat |
| 4 | New window | Stage 2b | Read the handoff, bring it to chat |
| 5 | New window | Stage 2c | Read the handoff, bring it to chat |
| 6 | New window | Stage 2d | Read the handoff, bring it to chat |
| 7 | New window | Stage 2e | Read the handoff, bring it to chat |
| 8 | New window | Stage 2f | Read the handoff, bring it to chat |
| 9 | New window | Stage 2g | Read the handoff, bring it to chat |
| 10 | New window | Stage 2h | Read the handoff, bring it to chat |
| 11 | - | Stage 2i - decided not to run | skipped |
| 12 | New window | Stage 2j - SENT AND RUN 2026-09-25 | `reports/STAGE_REPORT_2j.md` |
| 13 | New window | Stage 3 | Read the handoff, bring it to chat |
| 14 | New window | Stage 4 | Done |

Stage 2j runs, by the author's decision, and it runs BEFORE Stage 3 because it
adds a row to the scorecard figure.

The Project Brief is pasted once, in step 1. Stage 0's first instruction is
to save it to `CLAUDE.md` at the repository root, which Claude Code reads
automatically at the start of every session. From step 3 onward you paste
the stage text only. The pipeline roadmap is already in `CLAUDE.md` as of the
end of Stage 1, so every window inherits it without being given it again.

Stages 0 and 1 share a window because the refactor plan is built from the
inventory. Everything after that gets a fresh window, because a long
session accumulates detail that stops being relevant and starts being
noise.

Stages 2a, 2b and 2c must run in that order. They settle the data being
scored, how models are fit, and what they are scored against. Everything
from 2d onward assumes all three are done. 2a is the largest stage and may
take more than one sitting; that is fine, just stay in its window.

Each stage ends by writing a handoff file and telling you where it is.

### Sending results back to me for discussion

I work through findings and the manuscript in a separate Claude chat
window, which cannot see this repository. So at the end of every stage,
write a standalone handoff file to `reports/HANDOFF_stage-<id>.md` and
tell me explicitly that it is ready and where it is, so I can upload it.

That path and filename are the convention for every stage, matching
`reports/HANDOFF_stage-0.md` and `reports/MANUSCRIPT_discrepancies.md`. Do
not use any other directory or naming scheme, and if `CLAUDE.md` records a
different one from an earlier draft of this spec, correct `CLAUDE.md`.

**The handoff has exactly one audience and it has no access to this
repository.** The session that reads it works on the manuscript and drafts the
next stage's prompt from a single uploaded file. It cannot open `CLAUDE.md`,
`reports/MANUSCRIPT_discrepancies.md`, `CONTEXT.md`, any table, any figure, any
audit script, anything in `refs/`, or any source file. So:

- A `decision N` or `entry N` reference is a trailing citation and never the
  substance. If a sentence cannot be understood without looking the number up,
  spell it out in the handoff.
- Never instruct that reader to consult a repository file. "Take the numbers from
  the rebuilt tables" or "read the details off `refs/<paper>.pdf`" is an
  instruction it cannot follow. Either put the numbers and the details in the
  handoff, or address the instruction explicitly to the next Claude Code session,
  which can.
- Quote the numbers themselves, not where they live. Figures, table contents and
  file sizes have to appear as text.
- The handoff is the only channel. Do not assume anything from a previous
  handoff is still in that session's view; earlier handoffs are deleted from the
  repository as stages close, and the reader may not have the ones that remain.

Each handoff must stand alone for a reader with no access to the code:

1. What was asked, in one short paragraph.
2. What was found, as direct answers to the specific questions the stage
   posed, with numbers. Answer every question, including the ones where the
   answer is "this turned out not to matter."
3. Every number that changed relative to the pinned regression values, with
   the old value, the new value, and the reason.
4. Anything surprising, wrong, or ambiguous in my existing code or
   assumptions, stated plainly rather than diplomatically.
5. Decisions you made that I did not specify, and what the alternatives were.
6. What this implies for the manuscript: which claims are now stronger,
   which are weaker, which limitations in the Discussion can be retired, and
   any new limitation introduced.
7. Open questions for me.

Plain ASCII, no Unicode subscripts, small tables inline rather than
references to output files I would have to open separately. Assume the
reader is technical but has not seen the code.

---

## Project Brief (paste at the top of a new window when needed)

I have a Jupyter-notebook analysis for a paper comparing six uncertainty
quantification (UQ) methods for probabilistic LCA of building materials.
The six methods are the cross of three probability estimation methods
(normal fit, lognormal fit, kernel density estimation) with two weighting
schemes (uniform, variable).

Key vocabulary:

- **ECC**: embodied carbon coefficient, the embodied carbon emissions per
  unit mass or volume of a building material, in kg CO2e/kg or kg CO2e/m3.
- **ECC dataset**: the set of ECC values for one building material
  category, drawn from environmental product declarations (EPDs).
- **MUI**: material use intensity, mass of a material per unit floor area.
- **pLCA**: probabilistic LCA. Here, a linear combination of several
  material ECC distributions evaluated by Monte Carlo simulation.
- **W1**: Wasserstein-1 distance, the area between two CDFs. The current
  goodness-of-fit metric.
- **Variable weighting**: weighting each ECC value by the market share of
  the product it represents, instead of weighting all values equally.

What the analysis currently does:

1. Extracts 138 empirical ECC datasets from the EC3 database by
   MasterFormat category, cleans them, and normalizes each to a weighted
   mean of 1.0.
2. Computes about 10 statistical characteristics per dataset (coefficient
   of variation, entropy, skewness, kurtosis, mode count, dataset size,
   Shapiro-Wilk normality and lognormality, weight of statistical outliers,
   and W1 between the uniform-weighted and variable-weighted versions of
   the same dataset).
3. Generates 10,000 synthetic ECC datasets exhibiting statistical
   characteristics in the ranges observed empirically.
4. Assigns variable weights to each data point from a flat Dirichlet
   distribution, since real market share data is unavailable.
5. Applies all six UQ methods to every dataset and scores goodness-of-fit
   as W1 between each fitted model's CDF and the variable-weighted
   empirical CDF of the data.
6. Randomly partitions the 10,000 synthetic datasets into 2,500 disjoint
   groups of four, runs each group as a pLCA by Monte Carlo simulation
   (n = 10,000, MUI = 1.0 for all four) under each of the six UQ methods,
   and compares downstream results, principally "ECI Rank #1 Frequency,"
   the share of Monte Carlo iterations in which a given dataset is the
   largest contributor to the total.

The code works but was written by hand over a long period without AI
assistance. It is slow, repetitive, and not organized for reuse. The paper
is drafted and in revision; the code is cited as a public Zenodo deposit.
The target journal is Building and Environment.

Two of my own published papers are directly relevant and should be treated
as constraints on consistency, not just as citations:

- Torres and Srubar (2025), "Characterizing statistical uncertainty and
  variability of building material emissions in probabilistic whole-building
  life cycle assessment using kernel density estimation," Building and
  Environment 284, 113442. This introduced the KDE-based UQ method (KL1).
- Torres, Lupton, Marsh, Srubar and Allen (2026), "Using kernel density
  estimation and the Dirichlet distribution for uncertainty quantification
  of building material emissions," Resources, Conservation and Recycling
  234, 109022. This is KL2, which adds Dirichlet-sampled market-share
  weights, variable kernel bandwidths for product-level uncertainty, group
  weight constraints, and a representativeness parameter. Code at
  https://doi.org/10.5281/zenodo.19246153

Two papers by Ellen Marsh (University of Bath) are also directly relevant.
She is a co-author on the KL2 paper above and a collaborator from my
visiting appointment at Bath, but is not an author on the present paper:

- Marsh, Hattam and Allen (2025), Journal of Cleaner Production 491,
  144467. Market-share-weighted average LCIA results with uncertainty in
  both the impact data and the production volumes. Useful as a reference
  for realistic weight concentration: in their steel case, Rest-of-World
  BOF alone is 63.75 percent of global production while Austrian EAF is
  0.03 percent.
- Marsh, Lewis, Hattam and Allen (in press), "Uncertainty characterisation
  for construction products and comparative metrics for probabilistic
  building LCA." Six uncertainty characterisation scenarios applied to a
  four-option staircase comparison, evaluated with four comparative
  metrics. Two findings matter here: the ranking of top-contributing
  products within a design changes depending on the characterisation
  scenario, and they use dependent sampling across compared options.

Where the current analysis makes a choice that differs from those papers,
flag it rather than silently keeping either version. One is already known:
KL2 uses Silverman's rule for KDE bandwidth and justifies it explicitly,
while this analysis uses Scott's rule.

The manuscript currently cites the KL2 paper as "Torres et al. (in press)."
It is now published, as above. Update every such citation and check for any
other placeholder or in-press citations in the manuscript and code.

### Reference papers

PDFs are in `refs/`. Consult them where noted rather than working from
memory; several define methods this analysis implements directly. Stage 0
confirmed that every paper listed below is present, so no line needs
deleting.

Stage 0 also found three files in `refs/` that this manifest does not name.
Two of them matter and are now listed below: Silverman (1986) and a file
named `jcgs.2009.08054.pdf`. That second file is very likely Maitra and
Melnykov, "Simulating data to study performance of finite mixture modeling
and clustering algorithms," Journal of Computational and Graphical
Statistics, which is the paper Stage 2a Part 5 needs. Check its title page
before relying on it, and tell me if it turns out to be something else. The
third unlisted file, "The Embodied Carbon Benchmark Report 2025," is not
needed by any stage.

| File | Where it is needed |
|---|---|
| Torres and Srubar (2025), "Characterizing statistical uncertainty and variability of building material emissions in probabilistic whole-building life cycle assessment using kernel density estimation," Building and Environment 284, 113442 | Defines KL1, the KDE method under test |
| Torres et al. (2026), "Using kernel density estimation and the Dirichlet distribution for uncertainty quantification of building material emissions," RC&R 234, 109022 | Defines KL2: Dirichlet weighting, bandwidth with effective sample size and weighted IQR, group weight constraints, representativeness. Stages 2a, 2h |
| Melnykov, Chen and Maitra (2012), "MixSim: An R Package for Simulating Data to Study Performance of Clustering Algorithms," Journal of Statistical Software 51(12) | The software paper for the overlap parameterization. Stage 2a Part 5 |
| `refs/jcgs.2009.08054.pdf`, probably Maitra and Melnykov, "Simulating data to study performance of finite mixture modeling and clustering algorithms," JCGS 19(2) | The methods paper behind MixSim: the overlap parameterization itself and the solve-for-parameters algorithm. This is the one to implement from. Stage 2a Part 5 |
| Silverman (1986), "Density Estimation for Statistics and Data Analysis" | Source for the robust bandwidth rule (eq. 3.31) and for the critical-bandwidth mode test. Stages 2a Part 1, 2h |
| Marsh, Hattam and Allen (2025), "A method to create weighted-average life cycle impact assessment results for construction products, and enable filtering throughout the design process," JCP 491, 144467 | Real market-share concentration figures. Anchors the concentration sweep in Stage 2h |
| Marsh, Lewis, Hattam and Allen (in press), "Uncertainty characterisation for construction products and comparative metrics for probabilistic building LCA" | Dependent sampling practice and comparative metrics. Stages 2e, 2g |
| Henriksson et al. (2015), "Product carbon footprints and their uncertainties in comparative decision contexts," PLoS One 10(3), e0121221 | Dependent sampling in comparative LCA. Stage 2e |
| Heijungs (2021), "Selecting the best product alternative in a sea of uncertainty," IJLCA 26, 616-632 | Discernibility index and modified comparison index. Stage 2g companion metrics |
| Prado-Lopez et al. (2014), "Stochastic multi-attribute analysis (SMAA) as an interpretation method for comparative life-cycle assessment (LCA)," IJLCA 19, 405-416 | SMAA and overlap area, the main alternative to W1. Implement overlap area alongside W1 as a robustness check in Stage 2c so I can justify the choice in the paper |
| Benke et al. (2025) | Harmonized building LCA quantities. Only if Stage 2i runs |

Standing constraints for all work on this project:

- Plain ASCII in all output, code comments, and figure text. No Unicode
  subscripts, no Unicode multiplication sign, no typographic quotes or
  dashes. Write CO2, not the subscript form.
- Never silently change a result. If a change moves a number, stop and
  tell me which number, by how much, and why.
- Prefer explicit and readable over clever. I need to defend every line of
  this to a reviewer.
- Every analysis writes a tidy results table to disk. Figures are generated
  from those tables, never from in-memory state.
- All randomness comes from an explicitly passed Generator, never from
  global numpy state.

### Measurement is yours; framing is not

Every stage produces two kinds of output and they have different owners.

**Measurement.** What a quantity is, how it behaves, whether an effect survives a
control, which of two settings scores better on a stated criterion. This is
yours. Specify it, run it, and report it. A wrong measurement is catchable by
another measurement, which is why the stages so far have worked.

**Framing.** What a measurement means for a practitioner, what the paper should
claim from it, whether a correct result is actionable advice or only a
description. This is NOT yours, and Stage 2d is the evidence: its
location/shape split was measured correctly and then offered as a recipe a
practitioner could follow, which it is not, because nobody can compute a
market-weighted mean without already knowing the market shares. No test catches
that. It took three rounds of author correction, and the stage's two most useful
findings came out of the author's review rather than from the stage.

So, for the rest of the project:

- **Report measurements, and state the candidate readings rather than choosing
  between them.** If a result could support two claims of different strength,
  give both and say what would distinguish them. Do not pick the stronger one and
  write it as a finding.
- **Flag any sentence that tells a practitioner to do something.** Those are the
  ones that go wrong. For each, state explicitly what the practitioner would have
  to already know in order to follow it. If the answer includes anything the
  companion paper exists because nobody has, it is not advice and must not be
  written as advice.
- **Do not draft manuscript claims.** The handoff's job is to put the numbers and
  their limits in front of the manuscript session. That session and the author
  write the claims.
- **When an instruction is corrected, apply it to the whole project and not to
  the instance.** The figure style guide is binding on every figure in the
  repository, not on the figures the current stage produced. A correction about
  significant figures applies to every number in the handoff. If a correction has
  a scope you are unsure about, assume the wider one.
- **If the prompt specifies an instrument, a measure or a method and the data does
  not support it, stop and say so rather than delivering it.** Stage 2d was told
  to use A_IQR as its instrument and A_IQR turned out to be a function of dataset
  size at Spearman -0.946, for the structural reason that this study holds weight
  information fixed. The prompt was wrong. Reporting that early is worth more than
  completing the task as written.

### Reopening generation or the empirical arm

**A defect that is worth fixing gets fixed, whatever stage finds it.** There is no
freeze and no point past which a wrong input is preferred to a right one. What
there is, is a cost: the corpus and the empirical extract sit upstream of every
number in the study, so changing either means everything downstream re-runs, and a
change made casually mid-stage is how a study ends up unable to say which
configuration produced which result.

So the constraint is on process, not on permission:

- **Do not change either one inside a stage doing other work.** Finish the
  measurement that revealed the problem, report it, and let the change be its own
  scoped piece of work with its own handoff. Stages 2a-2 and 2a-3 were exactly
  that and both were worth their cost.
- **Bring the measurement, not the intuition.** State what is wrong, what the fix
  changes, and what it costs. Several proposed fixes in this project have been
  correct about the defect and rejected on the measured cost, which is a good
  outcome and only possible because the cost was measured first.
- **Say so before starting, not after.** If a stage concludes a rebuild is
  warranted, that is a recommendation to the author in the handoff, and the author
  decides. The reason is not that rebuilding is forbidden; it is that the author
  is the one who knows what else is in flight.
- **When a rebuild happens, everything downstream re-runs and the handoff says
  which published numbers moved.** No silent re-baselining.

A finding that does not clear that bar is still worth writing down, as a
sensitivity or as a sentence in the limitations. Most will be. But "it is too late
in the project" is never the reason.

### Version control: what is tracked, what is ignored

This repository is public and Zenodo-archived, and git history retains a file
even after a later deletion. Every stage below opens by telling you to commit
uncommitted work. That instruction means the code you wrote; it is not a
licence to run `git add -A` over the working tree. Check what you are staging.

Never tracked, under any circumstances:

- **`outputs/manuscript/`.** Already gitignored, and it stays that way. It
  holds the manuscript docx carrying unresolved tracked changes and comments
  from a named third party. Publishing a colleague's private editorial
  feedback is not recoverable by deleting the file later. If you find it
  committed on any branch, stop and tell me rather than rewriting history on
  your own.
- **`refs/`.** Gitignored in Stage 0: publisher PDFs and a third-party
  supplementary spreadsheet, about 177 MB, and a licensing problem besides.
  The files stay on disk for you to consult.
- The dead notebooks (`VOID_*`, `*_backup`, `_SCRATCHPAD`), already ignored.

Tracked, and must stay tracked:

- `environment.yml`, `tests/` including `tests/fixtures/`, everything under
  `src/` and `notebooks/`, `reports/`, and the results tables under
  `outputs/tables/`. The regression fixtures are the mechanism enforcing
  "never silently change a result," so they are worthless untracked.
- `archive/`, if Stage 3 creates it. Moving a stale figure there should be
  recorded as a rename, not as a deletion.

Two specific points, because both have a tempting wrong answer:

- **`data/processed/DATA_all.json` stays gitignored, but must stop being
  required.** It is 122 MB, so committing it is not the answer. Stage 0 found
  that a fresh clone can neither read it nor regenerate it, which is the
  reproducibility hole Phase 2 item 11 closes: it becomes regenerable from a
  recorded seed, and the seed and run metadata are what get tracked. A file
  that is both required and ignored must not survive Stage 1.
- **Do not strip notebook outputs.** No `nbstripout`, no output-clearing
  filter, no pre-commit hook that does it silently. Committed outputs are diff
  noise and I know it, but notebooks remain the entry point specifically
  because inputs and results are visible inline to an outside reader of the
  code, and stripping outputs removes the reason for that choice. If notebook
  size becomes a real problem, tell me and I will decide; do not solve it
  unilaterally.

For figures, once Stage 3 establishes the naming convention: track anything
matching the `CompareUQMethods_FIG<N>_*` or `CompareUQMethods_SUPP<N>_*`
pattern, since those are the cited deposit artifacts, and ignore everything
else under `outputs/figures/`. Exploratory multi-panel figures at high dpi are
regenerable and do not belong in history. Until that convention exists, leave
the figure directory's tracking as it is.

---

## Stage 0 - orientation

*Paste the Project Brief first. Stage 0 and Stage 1 share one window.*

Before anything else: commit any uncommitted work on the current branch
with a clear message, then create and check out a new branch named
`stage-0-1-refactor`. Do all of this stage's work there. Commit as you go, in
logical units, so I can bisect later if a number moves unexpectedly. Tell
me the branch name and the commit you branched from.

**First, before anything else, set up the project context file.** Copy the
Project Brief above verbatim into `CLAUDE.md` at the repository root,
including the vocabulary, the analysis summary, the reference manifest, and
the standing constraints. Add the handoff-file specification from the same
message. You read this file automatically at the start of every session, so
it means I never have to paste the brief again. Every later stage will add
to it. Tell me when it is written.

Then, without changing any analysis logic:

- Read every notebook and script and give me an inventory: what each one
  does, what it reads, what it writes, and roughly how long it takes.
- Build a dependency map showing the actual order things must run in.
- Flag duplicated logic, dead cells, hardcoded paths, and anything that
  depends on manual execution order or on state left in the kernel.
- Identify the top runtime bottlenecks and tell me what causes each
  (Python loops that should be vectorized, refitting inside loops,
  recomputing things that could be cached, single-threaded work that is
  embarrassingly parallel).
- Tell me where randomness enters and whether it is seeded.
- Read the synthetic ECC dataset generation code specifically, principally
  the data-creation notebook and `src/datageneration.py`
  (`random_irregular_dataset`, `generate_random_numbers`,
  `random_logcount`), and describe its algorithm in detail. I have specific
  questions about it in Stage 2a and I want your independent description
  before I ask them.
- Also inventory `src/customstats.py`, which holds the weighted statistics
  and fitting routines (`weighted_lognorm_fit`, `weighted_bw`,
  `wasserstein1_weighted`, `empirical_metadata`, `shapiro_wilk_weighted`
  and others). Most of the methodological questions in Stage 2 land here.
- Write the handoff file for this stage as described above, covering both
  the inventory and the coding assessment below.

Give me the inventory and a proposed refactor plan. Do not start
refactoring until I approve the plan.

**Separately, assess my coding.** This is low priority and purely for my
own information, so do it after the inventory, but do it honestly rather
than kindly. Assess the code as I wrote it, before any refactoring.

Rate me on: correctness and numerical care; code organization and reuse;
naming and readability; testing and validation habits; handling of
randomness and reproducibility; performance awareness; appropriate use of
the scientific Python stack (numpy, scipy, pandas vectorization idioms);
version control and project structure hygiene; and statistical
implementation judgment as distinct from statistical knowledge.

For each, cite specific lines or patterns as evidence rather than
generalizing. Then place me against archetypes: a first-year data science
undergraduate, an entry-level industry data scientist, a self-taught
researcher who codes to get results, a domain scientist with strong
computational training, a research software engineer. Say which I most
resemble and where I deviate from that profile in both directions.

Finish with the three habits that would most improve my code, ordered by
payoff. Do not soften the assessment; an inflated version is useless to me.

---

## Stage 1 - refactor with results locked

*Same window and same branch as Stage 0. Do not create a new branch.*

The plan in section 10 of `reports/HANDOFF_stage-0.md` is approved as written,
with the four amendments below. Execute it phase by phase. Read section 10 and
section 11.2 of that handoff before starting; this prompt does not restate the
plan, it only amends and constrains it.

### Amendments

**A1. NB2 cannot run from a clean kernel, so Phase 0 item 2 will fail as
written.** Cell 65 uses `metrics`, which is only ever assigned in cell 62,
which is entirely commented out. Before the baseline run, make the minimum
change needed to let NB2 execute top to bottom, as its own commit, labelled
NEUTRAL, with the change described. Then take the baseline. If the minimal
change turns out to be ambiguous, stop and ask rather than guessing what cell
62 was meant to do.

**A2. The Phase 1 pLCA table is an archive, not a regression fixture.** At
Phase 1 nothing is seeded, so the pLCA results cannot be reproduced by rerunning
and no regression test over them is meaningful. Capture the table and label it
explicitly as a record of the values the current manuscript reports. The real
pLCA regression fixture is created after Phase 2 establishes seeding, and
frozen again after the `neccs` change in Phase 3. Keep all three artifacts:
the pre-seeding archive at `neccs=1000`, the post-seeding fixture at
`neccs=1000`, and the post-seeding fixture at `neccs=10000`. The differences
between them are exactly what the handoff needs to report.

**A3. Normalization stays as the code has it, by the unweighted mean.** This
reverses decision 2 in the Stage 0 handoff. Reason: the practitioner-facing
threshold this study is building is of the form "if reweighting shifts a
material by more than X percent of its mean ECC, weighting matters more than
the choice of distribution." A practitioner can compute an unweighted mean from
a set of EPDs. They cannot compute the market-weighted mean without already
knowing the market shares, which is the quantity they lack and the reason the
rule exists. So no code change: `data / np.mean(data)` stays in both the
synthetic and empirical paths. The manuscript text is what is wrong, and that
goes in the discrepancy log below. This also removes normalization from the
Stage 2 regeneration list, so record it as resolved rather than carried
forward.

**A4. The `weighted_quantile` scope question is closed.** It is called only by
`weighted_bw`'s Silverman branch and by the dead `bw_dirichlet`.
`empirical_metadata` computes its quartiles independently through `np.interp`
on the weighted eCDF, so no reported metric is contaminated. Fix the function
and test it as planned under Phase 3 item 12, but do not spend time auditing
metric columns for downstream effects.

### Additional Phase 6 item: the manuscript discrepancy log

Create `reports/MANUSCRIPT_discrepancies.md`. It is the checklist the author
will work from when revising the paper, so each entry needs: what the
manuscript says, what the code does, and whether the fix is to the text or to
the analysis. Seed it with everything known so far:

- pLCA Monte Carlo sample size stated as 10,000, run at 1,000. Fixed in Stage 1
  by moving to 10,000, so the text becomes correct.
- Normalization described as a weighted mean of 1.0; computed from the
  unweighted mean. Per A3 the text is wrong and the code stands.
- "Mode Count" is a continuous modality index from `estimate_maxima`, not a
  count of modes.
- The 27.5 percent outlier filter is undescribed.
- Effective maximum dataset size is 749, not the stated 1,000.
- `logfit_offset = 0.5` is undocumented.
- The `(1-capecc)` divisor on cap rank frequencies is undocumented and carries
  an inline comment conceding the percentages look off.
- The `**exp` variance inflation in generation is undescribed.
- Nineteen statistical metrics stated, against 18 panels covering roughly 10
  distinct metrics.
- Scott's rule is used without justification, while the author's KL2 paper uses
  Silverman and justifies it explicitly.
- `fit_norm_SW` and `fit_norm_SW_uw` come from different estimators
  (Shapiro-Wilk and Shapiro-Francia) but are presented and discussed as a
  uniform-versus-variable comparison of one statistic.

Every later stage appends to this file.

### Constraints on this stage

- **Exactly two changes in Stage 1 may move a number**: `neccs` from 1,000 to
  10,000, and the `wbeci_mean` / `wbeci_stdev` assignment moving inside the
  per-dataset loop. Every other phase must leave the regression fixtures
  passing unchanged. A refactor that quietly moves a third number is the main
  failure mode here. If a fixture breaks in a phase labelled NEUTRAL, stop,
  find out why, and report it rather than re-baselining.
- **Phases 0 and 1 are gates.** No refactoring begins until the pinned
  environment, the regression fixtures and the persisted pLCA table exist.
- **Do not pull anything forward from section 10.9.** Those are Stage 2 items
  and taking one early forces a second regeneration of the datasets, which
  invalidates every number in the paper a second time for no gain.
- One commit per change that moves a number, with the before value, the after
  value and the reason in the commit message.

### At the end of Stage 1, extend CLAUDE.md

Stage 0 created it with the project brief. Add: the package layout; the
interface for each fitting method; the seeding and caching conventions; how to
run the pinned environment and the smoke configuration; the regression fixture
inventory and what each one pins; and a running log of decisions with their
reasons, newest last. Keep it current at the end of every subsequent stage, and
add the figure naming convention once Stage 3 establishes it. If it grows
unwieldy, move the stable reference material to `CONTEXT.md` and leave
`CLAUDE.md` as a short pointer plus the decision log.

Then write `reports/HANDOFF_stage-1.md` per the specification in CLAUDE.md,
opening with a "Carried forward" section that restates every still-open item
from the Stage 0 handoff, each marked resolved, still open, or superseded.

## Stage 2 - analysis extensions

*Each item below is its own window and its own branch.*

Run 2a, 2b and 2c in that order and in that order only. They change the
data being scored, how models are fit, and what they are scored against.
Everything from 2d onward has to run against settled versions of all three.

---

### Scope discipline across stages

The pipeline roadmap is in `CLAUDE.md`, already written there at the end of
Stage 1. Read it before starting, and apply its rule: if you hit something the
roadmap assigns to another stage, do not fix it, do not sketch a solution, and
do not run a quick check "just to know." Write one line in your handoff naming
what you saw and which stage owns it, then carry on with your own stage. The
failure mode is a stage that does its own work plus a thin version of three
others, leaving me unable to tell which number came from which decision.

Mark each stage done in the roadmap as it completes, and if a stage hands an
item to a different stage than the roadmap says, update the roadmap rather than
leaving the two out of step.

---

### Stage 2a - audit the synthetic dataset generator

Read `CLAUDE.md` at the repository root for project context before
starting. Then commit any uncommitted work on the current branch
with a clear message, then create and check out a new branch named
`stage-2a-generator`. Do all of this stage's work there. Commit as you go, in
logical units, so I can bisect later if a number moves unexpectedly. Tell
me the branch name and the commit you branched from.

The generator builds each synthetic dataset as a mixture, generatively,
with non-Gaussian components, so a parent distribution already exists for
every synthetic dataset. That is why the circularity problem is fixed by
changing the evaluation target in Stage 2c, not by changing generation.

**This stage nonetheless simplifies the generator, by decision, and the
instruction in an earlier version of this prompt not to rebuild it is
withdrawn.** The reason is not correctness but provenance: several steps
exist because they were tuned iteratively until the output looked right,
and each one makes the parent harder to write down. Part 1 below says which
steps go and what replaces them. The goal is a generator whose parent CDF is
closed-form, because Stage 2c scores against that parent, and a generator
whose every parameter comes from a configuration rather than from a tuning
history.

The target the simplification must hit: the synthetic corpus reproduces the
joint distribution of statistical characteristics of the 147 empirical
datasets, with enough margin beyond the empirical envelope to support a
generalizability claim. That is the point of the exercise. Part 6 measures
it, and if a simplification costs empirical coverage, report the tradeoff
rather than taking it silently.

The relevant code is in the data-creation notebook and in
`src/datageneration.py`, principally `random_irregular_dataset`,
`generate_random_numbers` and `random_logcount`. Read them before
answering anything below.

Expect to regenerate the datasets at the end of this stage. Several known
defects are listed below and fixing them changes every downstream number.
That is expected and acceptable; document each change against the pinned
baseline from Stage 1.

**Part 0: confirm and fix the known defects. Do this first.**

I have identified these by reading the code. Confirm each one empirically
before changing anything, quantify its effect, then fix it. Report the
before-and-after for each.

1. **The seeding collapse. Stage 1 already fixed the mechanism; what is left
   is quantifying the damage to the shipped data and deciding whether the
   algorithm needs more than correct seeding.** Do not re-thread the
   generators, and do not re-derive the defect from scratch. Stage 1 removed
   the `seed=0` default, made the Generator a required argument in every
   `src/datageneration.py` function, gave the lognormal branch its missing
   `random_state`, and verified that two same-length Gaussian components now
   differ by 3.36 in standardized units against about 1e-15 before. That is
   settled and tested.

   `DATA_all.json` was **not** regenerated, so the shipped datasets still carry
   the collapse: shape parameters constant across the entire study
   (skew-normal a = 3.458732, Student-t df = 3.366328, lognormal s = 1.136962),
   component samples deterministic given type and count, and only the lognormal
   branch genuinely random. Your job is to quantify how much that reduced the
   effective coverage of the metric space across the 10,000 shipped datasets,
   compare it against what the corrected generator produces, and then answer
   the question Stage 0 left open and Stage 1 could not: whether correct seeding
   alone is sufficient, or whether the component-shape parameters should be
   drawn per component rather than being constants at all. Report the
   before-and-after over the whole corpus, since the paper's headline
   percentages and any confidence interval across datasets assume independent
   draws.
2. **Dirichlet concentration mismatch.** Synthetic point weights use
   `np.random.dirichlet(np.ones_like(data))`, alpha = 1. Empirical weights
   in the data-creation notebook use `np.ones_like(data)*5`. The two arms of
   the study therefore received systematically different weight
   concentrations, on the exact dimension the paper is about. **Decided:
   alpha = 1 in both arms.** Change the empirical path from 5 to 1; do not
   change the synthetic path. Reasons, so you can reproduce the argument in
   the handoff: alpha = 1 is what the manuscript already describes, it is the
   maximum-entropy prior over unknown market shares, and it is the less
   concentrated of the two, which makes the measured weighting effect a lower
   bound rather than a fitted value. Report how much the
   empirical-versus-synthetic comparison changes once both arms agree. Stage 1
   already moved both lines off global numpy state, and Stage 2h sweeps
   concentration, so this only sets where the headline sits.
3. **Normalization, and the `+1` buffer goes.** `data = data + 1` then
   `data = data / np.mean(data)` normalizes by the *unweighted* mean, and the
   empirical path does the same. Per amendment A3 that is correct and stays;
   the manuscript text is what is wrong, and it is already logged.

   **Decided: remove the `+1`.** It is undocumented, it compresses the
   coefficient of variation, and it makes the two arms asymmetric, because
   synthetic values are held away from zero while empirical values are not.
   That asymmetry was how near-zero empirical values survived cleaning.
   The correct treatment is symmetric: no buffer on either side, and a
   multiplicative rather than additive low-end bound on the empirical
   cleaning step, so near-zero empirical values are removed on their merits.
   Fix the empirical bound here, since it is a generation-side decision about
   what the corpus contains; the consequences for fitting belong to Stage 2b.
   Report what removing the `+1` does to the achievable range of the
   coefficient of variation, because that is the metric it was distorting.
4. **Stale docstring.** The docstring for `random_irregular_dataset`
   describes 1 to 5 modes, Beta and Uniform components, and Pareto-style
   contamination. The code uses `rng.integers(1, 6)` with an inline comment
   saying 1-6, and the Beta, Uniform and contamination branches are
   commented out. Correct it. This is in a public Zenodo deposit. Stage 1
   reports having corrected some stale docstrings; check whether this one and
   `random_logcount`'s claim of a log-normal size distribution are already
   done, and if so confirm in one line rather than redoing them.

**Part 0b: regeneration hygiene. This is the one piece of Stage 1 plumbing
left unfinished, and this stage is where it closes.**

Phase 2 item 11 of the Stage 1 plan required that a file which is both
required and gitignored must not survive Stage 1. It did survive:
`data/processed/DATA_all.json` is 122 MB, gitignored, dated March 2026, and
was generated before seeding existed, so its current values can never be
reproduced. Every Stage 1 regression fixture is pinned to it. Your regeneration
is what finally closes that hole, so do it properly:

- **First action of this stage, before reading any code: freeze the baseline.**
  Copy the five pre-regeneration input files and the three pLCA artifacts named
  in `data/INPUTS.sha256` into `data/baseline_frozen/`, gitignored, then run
  `shasum -a 256 -c data/INPUTS.sha256` against both the originals and the
  copies and report both results. If either check fails, stop and tell me
  before doing anything else. This costs a couple of minutes and about 130 MB,
  and it means nothing this stage does can destroy the only existing record of
  what the manuscript currently reports.
- **Do not overwrite `data/processed/DATA_all.json`.** Write the regenerated
  corpus to a new, dated filename and repoint the notebooks at it. The existing
  file was generated in March 2026 before seeding existed and cannot be
  reproduced by any means, every Stage 1 regression fixture is pinned to it, and
  the repository sits in a synced folder, so an in-place overwrite propagates
  off-machine within seconds. Keeping both on disk removes the only
  irreversible step in this stage. Same for the three pLCA artifacts and
  the frozen fixtures: add, never replace.
- **`data/INPUTS.sha256` is append-only for its existing entries.** Stage 1
  created it to record the provenance and checksums of the pre-regeneration
  inputs. Add rows for the regenerated artifacts; do not rewrite or regenerate
  the file wholesale, and do not fold it into `SHA256SUMS.txt`. Those existing
  rows are what lets any surviving copy be proven to be the genuine
  pre-regeneration baseline. If the two manifest filenames overlap in scope,
  tell me and propose which is canonical rather than merging them yourself.
- Record the seed, the generator configuration and the code version in the run
  metadata written beside the regenerated data, so the new corpus is
  reproducible on demand from a tracked record rather than being an artifact
  that exists only on one disk. The data files themselves stay gitignored.
- Retire the `generate_dontread` flag. It is a hand-edited module-level boolean
  with no record in the outputs of which mode produced them, which is exactly
  the wrong mechanism for the one irreversible operation in this project.
  Replace it with something that records in the output what was done.

**Part 1: document what the generator does, then simplify it.**

First describe the algorithm precisely, as it stands: how the number of
modes is chosen, how component locations, scales and types are drawn, how
many points are drawn per mode, and how values are post-processed. Report
the actual distributions and ranges used for every parameter, read from the
code rather than inferred. This description is needed whether or not a step
survives, because the manuscript has to state what produced the old numbers.

Then apply the decisions below. Three steps are removed by decision, for
provenance rather than correctness: each exists because it was tuned until
the output looked right, and each one obstructs writing the parent CDF in
closed form. Quantify what each removal costs in empirical coverage before
and after, and if any removal loses coverage that Part 6 needs, say so
rather than proceeding quietly.

- **The truncation loop: replace it, do not characterize it.** `lo` and `hi`
  are computed once from the initial concatenated draw as
  `max(q1 - 3*iqr, 0)` and `q3 + 3*iqr`, and the `while` loop then enforces
  strict containment by refilling from a randomly chosen component until the
  dataset is exactly length n. That makes the parent a truncated mixture with
  conditional redrawing, reachable only by simulation. **Decided: sample the
  truncated mixture directly by inverse-CDF instead.** Exact, faster, and it
  removes a loop that can in principle fail to terminate. Report how much
  probability mass the truncation removes, since that still has to be stated,
  and confirm the two approaches agree distributionally before you delete the
  loop.
- **The power transform: remove it.** `exp = rng.uniform(0.9, 4.0)` then
  `data = data ** exp`. Its inline comment says the purpose is to align with
  empirical ECC data, which is to say it is a tuning knob for skew, and with
  locations up to 20 and exponents up to 4 it maps values as high as 160,000.
  **Decided: remove it and obtain skew directly from the component families
  in Part 5**, where it is a specified target rather than an induced side
  effect. Report the skew and kurtosis range achievable without it, and
  confirm against Part 6 that the empirical envelope is still covered. This
  one is the most likely of the three to cost coverage, so measure it before
  and after rather than assuming.
- **The reflection: remove it.** 25 percent of the time,
  `data = np.max(data) - data + np.min(data)`. It exists to produce
  left-skewed datasets, and it is the only sample-dependent transform in the
  generator, using the realized min and max rather than the parent's support
  bounds, which is precisely what makes the parent impossible to write down
  exactly. **Decided: remove it and get negative skew from the component
  families.** Confirm that the achievable negative-skew range still covers
  the empirical datasets that motivated it.
- **Component separation.** `locs` on (5, 20) with `scales` on (0.2, 1.5)
  places components roughly ten to fifty standard deviations apart, so
  multimodal datasets are well-separated clusters rather than the partially
  merged shoulders real material categories tend to produce. Quantify the
  actual pairwise overlap distribution and compare it against the 147
  empirical datasets. This is the strongest argument for the overlap
  parameterization in Part 5.
- **Mode counting.** The metric relies on `estimate_maxima`, which counts
  local maxima of something. Put it on a principled footing using
  Silverman's critical-bandwidth test: for a kernel density estimate with a
  Gaussian kernel, the number of modes is a non-increasing function of
  bandwidth, so the critical bandwidth for k modes is the smallest bandwidth
  at which the estimate has at most k modes, and significance is assessed by
  bootstrapping from the density smoothed at that critical bandwidth.
  Implement it, compare against the current mode count, and report how often
  they disagree. This also connects directly to the bandwidth question in
  Stage 2h.
- **Mode weight concentration.** `cpv = np.ones(k) * 10` gives a Dirichlet
  concentration of 10, so modes receive close to equal shares of points and
  mode dominance barely varies across the corpus. The commented-out
  alternative used a location-dependent concentration. Report the realized
  distribution of mode shares and whether it should be a swept parameter.
- **Dataset size: replaced by stratified sampling.** `random_logcount`
  currently draws log-uniformly on (4, 1000), and its docstring wrongly says
  normally distributed on a log scale. Confirm the realized distribution for
  the record, then replace it with the design below.

**Dataset size is now stratified, by decision.** The empirical datasets run
n = 3 to 77,548 with a median of 62, so a single log-uniform draw either
leaves the large regime too sparse to analyze or lets large datasets dominate
every aggregate. The design:

| Stratum | Size range | Datasets |
|---|---|---|
| 1 | 3 to 9 | 2,500 |
| 2 | 10 to 99 | 2,500 |
| 3 | 100 to 999 | 2,500 |
| 4 | 1,000 to 9,999 | 2,500 |

Total 10,000, which preserves the 2,500 disjoint pLCA groups of four. Draw
log-uniformly within each stratum. Equal allocation buys equal precision in
every size regime, which is what the metric-versus-W1 modeling in Stage 2f
needs.

Because equal allocation does not match the empirical size distribution,
**report every headline aggregate twice**: per stratum, which is the
scientific result, and reweighted by the empirical frequency of each stratum,
which is the number that describes real ECC datasets. Post-stratification
reweighting costs nothing and it is the direct answer to the objection that
the corpus over-represents large datasets.

**Plus a probe set, outside the corpus.** Generate about 50 datasets
log-uniform on 10,000 to 100,000, held separate from the 10,000 and excluded
from every aggregate. Their only job is to establish whether results have
plateaued by n = 10^4. Six empirical datasets exceed 9,999, reaching 77,548,
and they include the two largest and most carbon-significant materials, so the
coverage claim needs them addressed. If the probe set shows no further change
above 10^4, say so with the numbers and the claim closes; if it does show
change, that is a finding and we will revisit the stratum design.

Expect stratum 1 to produce undefined metrics: kurtosis divides by (n-3), so
n = 3 gives NaN, and normality statistics and KDE are degenerate at single
digits. That is expected and is itself informative, since small n is where
parametric families should beat KDE. Report how many metrics are undefined per
stratum, and flag it for Stage 2f, whose complete-case models will otherwise
silently drop the stratum.

Then write down the parent CDF explicitly. After the simplifications above it
is a composition of: the mixture, truncation with renormalization, and
division by the unweighted mean. That should be closed-form. Verify it
numerically anyway, by comparing a dense sample from the analytic parent
against a very large sample generated through the pipeline. Stage 2c scores
against this parent, so treat the verification as the deliverable of this
part.

**Part 2: the undocumented selection step, which is being dropped.**

After generating 15,000 datasets, the notebook removes every dataset that is
a marginal outlier on *any* metric, using Q1 - 1.5*IQR and Q3 + 1.5*IQR with
the IQR widened to `max(q3-q1, std*1.35)`, then keeps the first 10,000
survivors. It discards 27.5 percent of what was generated, it is keyed to the
synthetic corpus's own metric spread rather than to anything empirical, and it
is not described in the manuscript.

**Decided: drop it.** Three reasons to record in the handoff. It
preferentially removes high-`weight_outliers` datasets, which are the cases
the paper exists to study. Its cap on n at 749 is a side effect of filtering
on a metric that is now a design parameter rather than an outcome. And
fourteen datasets were being discarded for floating-point noise in a column
that is 1.0 by construction.

Replace it with a validity-only filter, which rejects a dataset solely
because it cannot be analyzed:

- Non-finite metrics that are not expected for the stratum. Undefined kurtosis
  at n = 3 is expected and must not cause rejection; see the stratum note in
  Part 1.
- Skewness and kurtosis combinations that are mathematically infeasible at
  that n, per Part 6.
- Degenerate weight vectors or zero-variance data.

Nothing is filtered on `weight_outliers`, and nothing is filtered for being
statistically unusual. Empirical plausibility becomes a *reported coverage
statistic* in Part 6 rather than an enforced criterion, which is the honest
version of the same idea: we say how much of the synthetic metric space is
empirically plausible instead of deleting the parts that are not.

Two things still to quantify, because the manuscript has to describe what the
old corpus was: what fraction the old filter removed per metric, and whether
the removed datasets shared a structural signature. Report the validity
filter's own rejection rate per stratum, which should be small, and flag it if
it is not, because a high rate means the generator is producing datasets it
should not.

**Part 3: where market share lives. This is the important one.**

The Dirichlet is used twice, independently:

1. `weights = np.random.dirichlet(np.ones(nmodes))` then
   `counts = np.random.multinomial(nvals, weights)`, which sets how many
   data points fall in each mode.
2. A second, separate Dirichlet draw assigns market-share weights to the
   realized data points.

Because those draws are uncoupled, the market-weighted distribution is
defined only on the realized sample: there is no population object for a
variable-weighted method to recover. The uniform-weighted methods are
estimating the generative mixture and can be scored against it; the
variable-weighted methods are estimating something with no parent. This is
the root of the circularity problem Stage 2c addresses.

Implement the fix: attach market share at the *mode* level, then distribute
each mode's share among that mode's own points. The market-weighted parent
then becomes a well-defined mixture, sum over k of v_k * f_k, where v_k are
market shares and the sample was drawn with mode counts from a separate
draw. Both weighting schemes then have a population to be right or wrong
about.

Make the coupling between mode membership and point weight a swept
parameter, with zero coupling (current behavior) at one end and fully
mode-determined at the other. Report how results depend on it. This also
aligns the study with the group weight constraints in the KL2 paper and
with the region-by-process-type weighting in Marsh et al. (2025).

**Part 4: reconcile code against manuscript.**

Two discrepancies I found; look for others.

- Empirical weights are drawn with `np.random.dirichlet(np.ones_like(data)*5)`,
  a concentration of 5, while the manuscript describes a uniform or flat
  Dirichlet. Report what concentration the synthetic datasets use. If the
  two arms differ, the empirical and synthetic results are not comparable on
  the dimension the paper is about.
- Normalization is **closed, do not re-open it.** Both paths divide by the
  unweighted mean, the manuscript text is what is wrong, and the code stands.
  That is Stage 1 amendment A3, reversing decision 2 in
  `reports/HANDOFF_stage-0.md` section 5, and it is already in
  `reports/MANUSCRIPT_discrepancies.md`. Confirm in one line that the synthetic
  and empirical paths still agree with each other after regeneration, and move
  on. The open question in Part 0 item 3 is only about the `+1` buffer.

**Part 5: replace the manual tweaking.**

Where generation parameters are hand-adjusted to make a dataset exhibit
desired characteristics, replace that with a swept configuration. Read
Maitra and Melnykov in JCGS, probably `refs/jcgs.2009.08054.pdf` per the
reference manifest, which parameterizes mixture simulation by pairwise
component overlap and solves for parameters achieving a specified overlap;
the MixSim paper in `refs/` is the R implementation of it and the
formulation ports over. Overlap is a
better generation parameter than the mode count I currently measure after
the fact. For shaping individual components, see moment-matching methods:
Fleishman (1978), Vale and Maurelli (1983), Headrick's extensions, the
Ramberg-Schmeiser generalized lambda distribution, the Johnson translation
system. These are unimodal families, so they shape components while the
mixture supplies multimodality.

Goal: every generation parameter set programmatically from a configuration,
no manual adjustment anywhere.

**Part 6: coverage, and Table 1 for the paper.**

- Report the joint distribution of statistical characteristics across the
  147 empirical datasets: correlation matrix plus a principal components or
  similar analysis of how many effectively independent dimensions exist.
- Do the same for the synthetic datasets and compare. Quantify how much of
  the synthetic metric space is empirically implausible: what fraction falls
  outside the empirical convex hull or in a low-density region of an
  empirical density estimate.
- Check for mathematically infeasible configurations. Kurtosis is bounded
  below by skewness squared plus one, and sample skewness and kurtosis are
  hard bounded by dataset size. Where a requested combination cannot exist
  at the requested size, report what the generator did. I suspect this
  explains an artifact at high negative kurtosis.
- Produce a table of every generation parameter with the empirical range it
  derives from and the synthetic range achieved. This becomes Table 1.
- Produce a coverage figure placing the 147 empirical datasets inside the
  synthetic cloud in the two or three most important metric dimensions,
  with a quantitative coverage statistic per metric, and a list of any
  empirical region the synthetic data fails to cover.
- **Report size coverage against the new strata.** The old corpus ran 4 to
  749 against an empirical 3 to 77,548, which is why the strata in Part 1
  exist. Confirm the new corpus spans 3 to 9,999 with the intended allocation,
  and report the probe set's verdict on whether results plateau above 10^4. If
  they do, state the coverage claim as: complete coverage to 9,999, with
  stability above that established on the probe set. If they do not, say so
  plainly, because then six empirical datasets sit in a regime the corpus does
  not represent and the claim has to be narrowed instead.

**Part 7: the empirical data at the source.**

The empirical metric distribution anchors all of the above. Check whether
the 138 EC3 datasets were deduplicated and whether industry-average EPDs
are mixed with product-specific ones in the same category. The extraction
already drops non-positive GWP values and trims at Q1 - 3*IQR and
Q3 + 3*IQR. Run a sensitivity check on those cleaning rules and report
how much the empirical metric ranges move.

---

### Stage 2a-2 - fresh empirical pull, and retune on it

Read `CLAUDE.md` and `reports/` in full before starting. Commit any
uncommitted work, then branch `stage-2a2-empirical` from `stage-2a-generator`.

**Where the generator's tuning rationale is recorded, so you do not rediscover
it.** This stage runs in a new session, and Stage 2a wrote its reasoning down
rather than leaving it in a conversation. Before changing any generation
parameter, read: every field docstring in `src/genconfig.py`, each of which
names the measurement that chose its value; section 4.3 of
`reports/HANDOFF_stage-2a.md`, which records three discarded corpora and the
cause of each failure; the multimodality section of that same handoff, which
explains why overlap was the wrong quantity to control at first and what
replaced it; and `audits/stage2a/b5_tune_configuration.py`, which is the
scoring loop you will be modifying. The operative facts are that
`position_skew` is the single strongest control on the coefficient of
variation, the overlap bounds are the control on modality, and the
coefficient-of-variation centre is a population target matched against a sample
characteristic, which is why it needed a 0.24 dex offset. Do not re-derive
these.

**This deliberately reopens generation.** Stage 2a's handoff says not to
regenerate, and that instruction is correct for every other stage. It is
suspended here, once, by decision, and it resumes the moment this stage
closes. The reason it is cheap right now: notebooks 2 and 3 have never been
run against `corpus_2026-09-12b`, so no downstream result exists that a
regeneration would invalidate. That stops being true as soon as Stage 2b runs.

Two changes, in this order, because the second depends on the first.

**Part 1: a fresh EC3 pull, cleaned symmetrically.**

The stored `dct_realeccs_trimmed.json` was already trimmed additively at the
high end when it was written, which is why Stage 2a could only apply a
multiplicative low-end bound. The cleaning rule is not cosmetic: Stage 2a
measured it moving `fit_norm_SW` by 1.55, entropy by 1.42 and `modality_index`
by 1.00 standard deviations. One arm of the study is carrying a cleaning rule
nobody chose.

- Read `../EPDsFromEC3/PULLING_EPDS.md` first, in full, including the three
  ways a paginated pull fails silently. Take a fresh pull.
- **Validate the pull before using it.** Record count per category, compared
  against the existing store's 206,668 records. A fresh pull should be a
  superset; if any category shrinks, that is a silent pagination failure until
  proven otherwise. Report the comparison rather than assuming it passed.
- **Freeze and checksum the raw pull** as a dated, immutable input, added to
  `data/INPUTS.sha256` alongside a note of the pull date and the API query. An
  API pull is not reproducible by a reader, so the archived file is what makes
  the empirical arm reproducible at all. This is the whole point of pulling
  rather than patching, so do not skip it.
- Apply the symmetric log-space cleaning rule to raw values, both ends.
- **Report which of the 138 categories change**: gained, lost, or crossed
  whatever inclusion threshold applies. If the count is no longer 138, say so
  prominently; the manuscript states it.
- Re-run the cleaning sensitivity analysis on the new data, since its previous
  result was measured on a file that had already been trimmed once.

**On the duplicate-manufacturer finding, decided so you do not have to
choose.** Stage 2a found 55 percent of records sharing a (manufacturer, GWP per
kg) pair, so the uniform-weighted empirical baseline is already implicitly
weighted by how many EPDs each manufacturer published. Keep EPD-level uniform
weighting as the primary analysis: it is what a practitioner pulling from EC3
actually holds, so it is the right baseline for a paper about what practitioners
should do. Build the deduplicated variant as a **sensitivity**, owned by Stage
2h, and state the implicit weighting explicitly in the text. Do not change the
primary definition here.

**Part 2: retune the corpus against the new empirical data, with modality
weighted.**

The corpus configuration was calibrated against the old empirical
characteristic distributions, so a fresh pull invalidates the calibration
whether or not anything else changes. Re-run
`audits/stage2a/b5_tune_configuration.py` against the new data.

Two changes to how it scores:

- **Weight the characteristics rather than averaging them.** The loop currently
  minimizes mean W1 across ten characteristics as if they were equally
  important. They are not. Modality and the coefficient of variation drive the
  KDE-versus-parametric comparison directly, because a lognormal cannot
  represent bimodality and a KDE can; `fit_norm_SW` and the rest are largely
  downstream of those two. Weight modality and coefficient of variation above
  the others. Report the weights you used and the score both weighted and
  unweighted, so the choice is visible rather than buried.
- **Close the modality gap if it can be closed cheaply.** The current corpus is
  88.4 percent unimodal against an empirical 81.9. Nudging the overlap lower
  bound down is the known lever. The target is matching the empirical modality
  distribution, not beating a threshold.

**This is bounded. Do not tune indefinitely.** If modality cannot be brought
within about 2 points of empirical without materially degrading the
coefficient of variation or skewness, stop, keep the better of the candidates,
and report the tradeoff with numbers. A corpus slightly too unimodal is
acceptable and its bias direction is known: it understates how often KDE should
win. What is not acceptable is trading a real characteristic away to chase
modality, or tuning until something looks right without recording why.

Test at 300 datasets before running 10,000, per the standing lesson in Stage
2a's handoff. One full run there was launched with no prior check and wasted 32
minutes.

Then regenerate once, into a new dated corpus directory, and repoint
`CORPUS.json`. Do not overwrite `corpus_2026-09-12b`; it stays on disk as the
comparison. Re-run notebook 1 end to end and report the coverage and modality
comparisons against the new empirical data.

**Acceptance, and what to report.** The new corpus matches the new empirical
data on the characteristic distributions at least as well as
`corpus_2026-09-12b` matched the old, with modality closer. Report both corpora
against both empirical datasets, four combinations, so it is possible to see
how much of any change came from the pull and how much from the retune.
Anything that got worse is reported, not smoothed over.

**Out of scope.** No fitting-method work, no lognormal, no pLCA. Notebooks 2
and 3 stay unrun; they are Stage 2b's first task. If something in this stage
looks like a Stage 2b question, write it in the handoff and leave it.

---

### Stage 2a-3 - split the heterogeneous categories, and check the envelope

Read `CLAUDE.md`, `CONTEXT.md` and `reports/` in full before starting, including
`reports/HANDOFF_stage-2a2.md` sections 3.3, 3.4 and 5. Commit any uncommitted
work, then branch `stage-2a3-categories` from `stage-2a2-empirical`.

**This is the last pre-2b stage.** Nothing after it reopens generation or the
empirical extract; see the standing constraint on pre-stages. Keep it short.

**Before touching generation, read section 7 of `reports/HANDOFF_stage-2a2.md`.**
Two errors accounted for most of that stage's length and both are easy to
repeat: tune against visible modes rather than Silverman, and keep the two arms
identical in every operation. If you find yourself tuning the generator to
compensate for something, check first whether the two arms differ.

**Scope. This is a short stage with one question and a conditional second
half.** Some EC3 categories are not a single product population. The author's
decision is to treat them as the separate populations they are, in the primary
analysis, provided each split can be substantiated. Your job is to substantiate
and apply the splits, measure what they do to the empirical envelope, and then
determine whether the generator needs recalibrating. Generation is reopened only
if the measurement says so, and the criterion is below.

**Part 1: the constraint that decides whether this is publishable.**

**Split only on record metadata. Never on the ECC values themselves.** A
category may be split using the declared unit type, the EC3 category path or
subcategory, a material or product-type field, or any other attribute carried on
the EPD record. It may not be split by clustering the ECC values, by looking at
where the density has a gap, or by any procedure that reads the distribution
being measured.

The reason is that this study measures modality, dispersion and skewness of ECC
distributions, and Stage 2a-2 found that 94.9 percent of empirical datasets have
a single visible mode. Splitting a category because its values look bimodal and
then reporting that empirical datasets are unimodal is circular, and a reviewer
will see it immediately. Splitting on the declared unit and then observing that
the resulting populations are less dispersed is a finding.

If a category is obviously heterogeneous but no metadata field separates it,
say so and leave it unsplit. That is an acceptable outcome and it is more
defensible than an unprincipled split.

**Part 2: choose what to split by a stated screen, not case by case.**

Define the screen before looking at results, and apply it to all 136
categories rather than to the three already named. Something like: more than one
declared unit type present, or coefficient of variation above a stated
threshold, or more than one EC3 subcategory represented. Report the screen, how
many categories it selects, and every category it selects, including any you
then could not split.

Stage 2a-2 already found the strongest signal for this: seven of the eight
categories whose medians disagree with the 2026-03 extract have more than one
declared unit type, and `Chairs` reads as `item` in the new extract where it was
mass in the old. Declared unit type is therefore the first axis to check, and
`outputs/tables/stage2a2/TABLE_2a2_ChangeDiagnosis.csv` already holds much of
the evidence.

Each resulting split needs one sentence of substantiation that could appear in
the paper, naming the field and the distinction. Produce a table of every split:
original category, field used, resulting populations, record counts, and the
one-sentence justification. Subcategories below the inclusion threshold of 3 are
dropped as before, and the new category count is reported prominently, since it
replaces 136 and the manuscript states it.

**Part 3: measure the envelope, then decide about regeneration.**

Re-run `audits/stage2a2/p6_empirical_envelope.py` and `p7_empirical_overlap.py`
on the split arm. Report every quantity that `src/genconfig.py` cites, before
and against after, in particular the median and maximum coefficient of
variation, the log10 standard deviation of it, median skewness and kurtosis,
median dataset size, the visible-mode distribution and the Silverman
distribution with `nboot` quoted.

**The criterion for reopening generation.** Compare the movement against the
seed-to-seed noise measured in `audits/stage2a2/p10_config_noise.py`: objective
standard deviation 0.0066, mode-count total variation 0.029. Score the current
corpus against the split empirical arm using the existing tuning objective, with
every characteristic weighted equally as Stage 2a-2 settled.

- If the corpus still matches the split arm within noise, **do not regenerate**.
  Report the four-way comparison and stop. This is the expected outcome if the
  splits touch few enough datasets.
- If it does not, re-run the tuning loop, adjust only the `genconfig.py` fields
  whose cited measurement actually moved, regenerate once into a new dated
  corpus, and report both corpora against both arms as Stage 2a-2 did. Do not
  retune fields whose measurement did not move.

Draft at 1,000 datasets before any full run. Test the split logic on the three
named categories before applying the screen to all 136.

**Part 4: two record corrections, both one-liners.**

- Discrepancy entry 30 says direct EC3 API access is closed to this account.
  It is not. The key in use at the time was stale; a working key has since been
  set. Correct the entry to say so. Do not take a pull.
- Decision 31 raises whether to fold in a newer EC3 pull. **Decided: no.** The
  empirical arm is frozen at the 2026-08 extract for the remainder of the
  project. It is archived, checksummed and reproducible from a clean clone, and
  one further month of declarations will not move the characteristic
  distributions enough to justify re-invalidating the calibration. Record the
  decision and the reason. If the arm is refreshed at all it will be once,
  deliberately, before submission, as an author decision.

**Out of scope.** No fitting-method work, no lognormal, no pLCA, and notebooks 2
and 3 stay unrun. They are Stage 2b's first task.

---

### Stage 2b - fitting methods: the lognormal, and fitting by the scoring criterion

**Four tasks before the lognormal work, in this order.**

**Before anything: which parts of `reports/HANDOFF_stage-2a3.md` are current.**
That stage tried three approaches and kept the third, and the document retains
the record of all three. Section 0, section 3.1c, section 4.3, section 4.3a and
section 7 describe the state that stands. Sections 3.2 through 3.8, section 4.4,
and the coverage entry in section 5 describe the withdrawn declared-unit split,
a 142-dataset arm and `corpus_2026-09-14b`, none of which exist any more. When
any two sources disagree, the canonical numbers block in
`reports/MANUSCRIPT_discrepancies.md` wins: **149 datasets, 117,090 values,
`corpus_2026-09-14d` at 9,999 datasets plus a 50-dataset probe set.**

0. **Remove physically implausible records, and report the extremes you cannot
   rule on.** This is an author decision from 2026-09-14 and it is the only
   change to the empirical arm that this stage makes. Do it before the notebooks
   run, and commit the re-frozen empirical fixture together with task 2's freeze
   rather than separately.

   The known case: `Elevators` contains three EPDs declared per kilogram
   reporting 20,812 kgCO2e/kg. That is not a product, it is a declaration error
   of about three orders of magnitude, and it currently drives both that
   category's dispersion and one of the coverage failures in discrepancy entry
   34.

   **The rule must be external, not distributional.** Do not set a bound from
   the arm's own quantiles, standard deviations or visible gaps. This study
   measures dispersion, so a bound read off the distribution would be circular
   in the same way a dispersion-based split was. Take the bound from material
   science: published embodied-carbon ranges for building products, for instance
   the ICE database or EC3's own documented ranges. Cite what you use in the
   docstring and in the split table, because the paper has to state it.

   **Apply it only where an external bound exists, which is the mass-declared
   categories.** No building product exceeds roughly 25 kgCO2e per kg; a ceiling
   at 100 is generous and still catches the Elevators records by two orders of
   magnitude. For volume, area, length and item declarations there is no
   comparably tight bound, so do not invent one. Instead, report the ten highest
   and ten lowest records per unit type with their product names and their ratio
   to their category median, as a table for author review. That is the check for
   other materials, and it is a report rather than a filter.

   **A gate, because the size of the catch tells you whether the rule is
   right.** If the ceiling removes more than about 0.1 percent of the arm's
   117,090 values, stop and report rather than proceeding. A handful of records
   means a declaration error; hundreds means the bound is wrong or the problem is
   something else. Report the count dropped, which categories lost records, and
   the effect on those categories' characteristics.

   **Then check whether the calibration moved**, using the existing criterion:
   score `corpus_2026-09-14d` against the arm before and after, against the
   seed-to-seed noise of 0.0066. Expect it to be well inside noise for a handful
   of records. **Do not regenerate.** If it somehow exceeds noise, stop and
   report; reopening generation is an author decision, not this stage's.

   **One count to report for the manuscript while you are in there.** The
   extraction keeps only records with GWP above zero. Report how many records
   that excludes, because some biobased products are legitimately
   carbon-negative, and an empirical arm truncated at zero by construction is
   worth a sentence in the paper alongside the (0, inf) support decision. This
   is a count, not a change: do not alter the filter.

1. **Run notebooks 2 and 3 against `corpus_2026-09-14d`**, which
   `data/processed/CORPUS.json` already points at. They have never been run
   against any final corpus, so the pipeline is unverified end to end past
   notebook 1. This is the oldest outstanding item in the project.

   **The corpus holds 9,999 datasets, not 10,000**, because one parent failed to
   solve and was reported rather than approximated. That breaks the 2,500
   disjoint pLCA groups of four exactly. Report how the remainder is handled
   rather than letting it fall out of an integer division: state the number of
   groups, the size of the last one, and whether any dataset is excluded. If the
   cleanest answer is to hold one dataset out of the pLCA grouping, do that and
   say which.

   Also recompute `EMPIRICAL_STRATUM_SHARE`. It currently reads 15/81/40/6 over
   142 datasets, which was measured on the withdrawn arm. It is a
   post-stratification weight and never a generation parameter, so updating it
   does not reopen generation. Use `COMPAREUQ_SMOKE_COMBOS=20` on notebook 3 first. Expect both
   to be slower than their Stage 1 timings, since stratum 4 reaches n = 9,996
   where the old corpus stopped at 749.

   **This is a gate.** If either notebook reveals a problem in the *generated
   corpus* rather than in the notebook, stop and report it. Do not fix it by
   changing generation, and do not regenerate: Stage 2a is closed and its
   configuration was arrived at by measurement against the empirical data. A
   generation defect found here means reopening 2a deliberately, which is my
   call, not a patch inside 2b.
2. **Re-freeze both metric fixtures against the active corpus.** The empirical
   fixture was re-frozen mid-way through Stage 2a-2, at commit `1b45ca2`, and
   four number-moving commits landed after it, so verify rather than assume: if
   `TABLE_EmpiricalECCMetricsAndW1.xlsx` or
   `TABLE_SyntheticECCMetricsAndW1.xlsx` does not reproduce from the current
   code and `corpus_2026-09-14d`, re-freeze it and record the deltas in the
   commit message. Then update `tests/fixtures/SHA256SUMS.txt`. Also drop
   `CORPUS.json` from `data/INPUTS.sha256`: it is a pointer to the active
   corpus, so pinning it guarantees a permanent failure and trains everyone to
   ignore the check.

**Use draft corpora while exploring.** `python corpus.py <label> 1000` builds in
about 80 seconds against 850, the label carries `draft1k` and `runmeta.json`
carries `n_corpus`, so a draft cannot be mistaken for a paper corpus. Stage
2a-2 measured a 1,000-dataset draft at mean W1 0.270 against the full corpus's
0.275, so drafts are representative. Confirm any conclusion on the full corpus
before it goes in a handoff.
3. **Mark decision log entry 13 confirmed in `CLAUDE.md`.** The author has
   confirmed it: ECC support is (0, inf), open at zero. It is no longer an open
   item and should stop being carried forward as one.

**Sort out the lognormal.**

Read `CLAUDE.md` at the repository root for project context before
starting. Then commit any uncommitted work on the current branch
with a clear message, then create and check out a new branch named
`stage-2b-fitting`. Do all of this stage's work there. Commit as you go, in
logical units, so I can bisect later if a number moves unexpectedly. Tell
me the branch name and the commit you branched from.

The three-parameter lognormal is currently fit by MLE with the data offset
by +0.5. That offset is suspicious given that all analysis values are
strictly positive and every dataset is normalized to a mean near 1, since a
two-parameter lognormal is perfectly well defined on strictly positive
data. Determine why the offset was needed.

**Near-zero values are much rarer than they were, but they have not gone
away.** Earlier versions of this prompt said the data contained no values at or
near zero. That was wrong at the time: the low-end bound was additive and never
bound, so the empirical arm kept values at 1e-17 of their mean, and the `+0.5`
offset was plausibly patching that rather than the threshold pathology.

Stage 2a-2 changed the picture. Both arms are now cleaned by the same
multiplicative log-space rule, which removed 816 of 120,280 empirical values,
including `Cement` at 4.7e-10 of its mean and `Asphalt` at 2.6e-08. But the
rule is permissive on a category whose log-space IQR is genuinely enormous, and
`Elevators` still retains a value at 1.2e-07 of its mean. A value that small
logs to about -16 against neighbours near 0, which is still enough to
destabilize a log-space fit.

So test both explanations, as before but on the current data: run the no-offset
two-parameter fit on the empirical datasets with and without the surviving
near-zero values, and report which of the two problems the offset was actually
solving, and on how many datasets. Do not change the cleaning rule; it is
settled and both arms depend on it being identical.

There is also a genuine pathology underneath, and I am stating it here
rather than pointing you at a paywalled 1963 paper. For the
three-parameter lognormal with threshold (location) parameter gamma, shape
sigma and scale mu, the likelihood is unbounded: as gamma approaches the
smallest observation from below, one term of the density diverges while the
others stay bounded, so the likelihood goes to infinity and the global MLE
does not exist. Any naive optimizer will either run gamma up against
min(x), collapse sigma toward zero, or return whatever the optimizer
happened to stop at. That is almost certainly what produced the weird fits
the +0.5 offset was patching.

The standard treatments are: (a) restrict gamma to a closed interval
bounded strictly below min(x) and take the interior local maximum of the
profile likelihood, which is what I want implemented; (b) a modified MLE
that replaces the score equation for gamma with a condition tying the
smallest order statistic to its expected value under the fitted
distribution; or (c) avoid the threshold parameter entirely by fitting the
two-parameter form to strictly positive data.

Note the Methods text is self-contradictory: it says shape, location and
scale are all estimated, then says location is held at zero. Determine what
the code does and tell me.

**Support is settled: every method lives on (0, inf), open at zero.**

This resolves decision log entry 13 in `CLAUDE.md`, which Stage 1 flagged for
confirmation. It is confirmed by the author, and it fixes the inconsistency
Stage 1 found where the scoring grid included zero while the sampler excluded
it. ECCs of exactly zero are not admissible, so zero is excluded rather than
included, and the grid is what changes, not the sampler.

The consequence is that the normal fit and the KDE must be **truncated to
(0, inf) and renormalized**, in scoring and in sampling alike. Both put
probability mass below zero: the normal directly, the KDE through Gaussian
kernels on small values. At present each is scored as an untruncated object
and then used as a truncated one, because `eccs[eccs > 0]` discards the
negative draws in the pLCA. That makes them one model when judged and another
when applied, which is not defensible.

Implement the renormalization once, in `src/fitting.py`, so all six methods
expose the same interface on the same support:

- The model CDF used for W1 is the truncated, renormalized CDF on (0, inf).
- Sampling is by inverse-CDF on that same truncated CDF, not by rejection.
  Rejection sampling gives the same distribution but wastes draws and, more
  importantly, breaks the common-random-numbers scheme Stage 2e installs,
  which needs one uniform variate per material per iteration to map through
  every method's inverse CDF.
- Lognormal and gamma are already on (0, inf) and need no change.
- Report how much the renormalization improves the normal fit's W1. It should
  improve, because the model stops being charged for mass in a region it is
  never allowed to occupy. That is the correct comparison and it is the one
  that answers the objection that the parametric families were set up to lose.

Report the grid used for W1 evaluation explicitly, with its lower bound
strictly above zero, and state how the bound was chosen.

Implement and compare:

- The two-parameter lognormal fit directly to the positive data, with no
  offset. If this works, it is the simplest answer and the offset goes away.
- Profile likelihood over the threshold: grid it over a bounded interval
  below the minimum observation, maximize remaining parameters at each grid
  point, take the interior local maximum. Standard treatment for this
  pathology.
- Gamma, which supports strictly positive data, has no threshold pathology,
  and is arguably the more natural competitor to lognormal for skewed
  positive data.

Report how much the offset was actually affecting results, so I know
whether this was a real problem or cosmetic.

**Fit by the criterion you score by.**

Every parametric family is fit by maximum likelihood but scored by W1. That
mismatch means the parametric families may never have had a fair shot under
the criterion used to judge them, which is a real vulnerability. Implement
direct W1 minimization as an alternative fitting method for every
parametric family, and report results under both MLE and W1-optimal fits.
If KDE still outperforms W1-optimally-fitted parametric families, the
finding is far stronger. If it does not, I need to know before a reviewer
tells me.

**Normalization: keep it exactly as it is.**

Every dataset is rescaled so that its *unweighted* mean is 1.0 before
fitting, in both the synthetic and empirical paths. Keep this, and do not
change it to the weighted mean.

Two reasons, and the second one supersedes an earlier decision. The first is
scale: the datasets are synthetic and unitless, and putting them on a common
scale is deliberate, because if one "material" sat at 10 while others sat at 1
it would dominate the pLCA regardless of distributional detail and the study
could not detect what it exists to detect. Same logic as MUI equal to 1.0 for
all materials. The second is that the normalizer must be computable by a
practitioner: the threshold this study is building has the form "if reweighting
shifts a material by more than X percent of its mean ECC, weighting matters
more than the choice of distribution," and a practitioner can compute an
unweighted mean from a set of EPDs but cannot compute the market-weighted mean
without already knowing the market shares, which is the quantity they lack and
the reason the rule exists.

This is Stage 1 amendment A3, which reverses decision 2 in
`reports/HANDOFF_stage-0.md` section 5. Treat A3 as current. The manuscript
text, which describes normalization as being to a weighted mean of 1.0, is what
is wrong; that is already logged in `reports/MANUSCRIPT_discrepancies.md` and
needs no further investigation here.

One consequence worth noting, because it removes work: since the unweighted
mean does not depend on the weight vector, the Stage 2h sweeps over Dirichlet
concentration and over multiple weight realizations cannot shift the normalizer
underneath the comparison. No special handling is needed there.

---

### Stage 2c - fix the evaluation target

**Read `reports/HANDOFF_stage-2b.md` sections 4.11, 4.12, 4.13, 4.14 and 4.15
before anything else.** They are why this stage now carries the paper's central
claim rather than a methodological cleanup. One internal inconsistency to know
about: the "extremes that survive every filter" passage says the
`THHN/THWN-2` record is left in the arm; section 4.20 and decision 63 removed it
and two records went. Section 4.20 is the later state. The arm is **147 datasets
and 116,766 values**.

**Task 0: adopt the guarded Silverman bandwidth, then re-examine it against the
parent.**

`BW_METHOD` moves from `'scott'` to `'silverman_guarded'` with
`SILVERMAN_MIN_NEFF = 20.0`. The rule as it runs, stated exactly because it has
been described wrongly twice: `n_eff` is Kish's effective sample size on the
normalized weights; `scale` is `min(std, iqr/1.34)` when `n_eff >= 20.0` and
`std` otherwise; `bw = 0.9 * scale * n_eff**-0.2`. **The guard swaps the scale
estimate only, never the coefficient**, which stays 0.9 on both sides where
Scott's is 1.06, so this is not a fallback to Scott and calling it one overstates
the small-sample bandwidth by 18 percent. The threshold is on `n_eff`, not on
`n`. There is a second, separate guard: a weighted IQR that is zero or non-finite
is set to `std * 1.34`, which makes the minimum equal `std` whatever `n_eff` is. This is an author decision taken on the strength of
Stage 2b section 4.12, and it is adopted here rather than in Stage 2h because it
changes which method wins, so every stage downstream of it would otherwise be
analyzing an answer that later moves.

Why it is defensible, which the paper has to say in this order:

- Scott oversmooths this data. Its median bandwidth is 0.56 of the data's
  standard deviation on the empirical arm, which inflates the fitted variance by
  about 15 percent and puts a mean of 9.4 percent of the fitted KDE's mass below
  zero on an arm whose support is (0, inf).
- Silverman's robust rule is what Torres et al. (2026), the author's own KL2
  paper, uses and justifies. Decision 9 and discrepancy entry 10 have recorded
  that inconsistency since Stage 0.
- Silverman's known small-sample failure was diagnosed and fixed. It is not the
  low `IQR/sd` cases, where Silverman beats Scott on held-out likelihood 100
  percent of the time; it is small effective sample size, where the interquartile
  range is interpolated between two order statistics. The guard is a minimum Kish
  effective sample size of 20, not a floor on the scale, and flooring the scale
  makes things worse.
- **The guard was calibrated on leave-one-out likelihood, not on W1.** That is
  the whole defensibility argument and it must survive into the text: the
  threshold is not tuned to the criterion the study scores by. W1 still prefers
  pure Silverman, and that is W1's undersmoothing bias rather than evidence.

**Then test it against a better referee than the one that chose it.** Leave-one-
out likelihood was used because W1 cannot arbitrate a bandwidth: W1 falls
monotonically as the bandwidth shrinks, minimizing at 0.02 of Scott's for 95 of
147 datasets, because a KDE with a vanishing bandwidth is the empirical
distribution it is being scored against. This stage installs the one referee with
no such bias, the known parent, so re-run the bandwidth comparison against
`MixtureParent.cdf(scheme='market')` and report whether it confirms the guarded
rule. If it does not, say so plainly and stop rather than adjusting the guard;
changing it again is an author decision.

**Report Scott as a sensitivity throughout, not as a discarded option.** The
sequence matters for the paper's credibility: the KDE lost under Scott, the loss
was investigated, and Scott turned out to be indefensible. A reader must be able
to see that the bandwidth was not chosen to produce the conclusion.

**Task 0b: apply post-stratification to the W1 results.** Stage 2b found that the
corpus holds 25 percent of its datasets above n = 1,000 against the empirical
arm's 7.4 percent, and that reweighting the corpus to the empirical size mix
makes the two arms agree on mean rank where they appeared to disagree. Equal
allocation across strata is a precision choice, not a claim about how common each
size is. `coverage.post_stratified` already exists and no stage has applied it to
the W1 or rank results. **Every headline aggregate in this stage and after is
reported both ways**, equal-allocation and empirically reweighted.

**Task 0c: two framing constraints that come out of the weight-draw noise
floor.** Stage 2b measured W1 between the same values under two independent
Dirichlet draws at a median of 0.1344 on the empirical arm, against the best
method's median W1 of 0.0984. The target's own noise exceeds the best score.

- **State the empirical headline as a win share, not as a mean rank.** Over five
  weight realizations `KDE, Variable` and `Lognormal, Variable` are within the
  draw noise of each other on mean rank, 2.243 against 2.353, with the lognormal
  ahead in one of five; on win share the KDE leads in every realization, 39 to 45
  percent against 25 to 31.
- **Make no size-banded claim below about n = 100** without the relative-gap view
  beside it. In that band the noise floor of 0.15 to 0.21 is the same size as the
  entire spread between methods, and at n = 3-9 the profile lognormal sits at its
  normal limit in 40 percent of fits and at its guard in another 40, so two of the
  three methods are frequently the same object.

**Task 0d: answer the gamma question rather than leaving it to a reviewer.** The
three-parameter lognormal's threshold is set by a guard rather than by the data
for about half the empirical arm. Gamma has no threshold, no pathology and no
guard, and beats the lognormal on the guard-bound datasets. The out-of-sample
comparison below is what should settle whether the three-parameter lognormal
earns its place. If it does not, say so; a hybrid estimator is the option the
measurements least favour, and if one is wanted, gamma is the fallback.

**Task 0e: W1 is nearly blind to tail mass and the pLCA is not.** Stage 2b
section 4.9 produced fitted models with standard deviations in the thousands on
data whose own standard deviation is 0.6, and every W1 score looked fine; it
surfaced in the pLCA results. This stage owns the evaluation target, so decide
whether W1 alone is sufficient, or whether a tail-sensitive companion belongs
beside it. Stage 2g inherits the same question from the downstream end.



Read `CLAUDE.md` at the repository root for project context before
starting. Then commit any uncommitted work on the current branch
with a clear message, then create and check out a new branch named
`stage-2c-target`. Do all of this stage's work there. Commit as you go, in
logical units, so I can bisect later if a number moves unexpectedly. Tell
me the branch name and the commit you branched from.

Every model is currently scored by W1 against the variable-weighted
empirical CDF of the same data it was fit to. That structurally favors KDE,
a flexible method scored on its own training data, and it makes "variable
weighting improves fit" close to true by construction, since the
variable-weighted eCDF is itself the target.

The fix differs for the synthetic and empirical datasets. Do both.

**Synthetic: score against the known parent.** Stage 2a establishes that
the generative mixture is recoverable and, in Part 3, gives the
variable-weighted methods a parent too. The primary score becomes W1
between each fitted model's CDF and the appropriate parent CDF, evaluated
by dense sampling if there is no closed form: the sampling mixture for
uniform-weighted methods, the market-weighted mixture for variable-weighted
methods. This is a standard estimator-recovery design. It removes the
circularity and penalizes overfitting automatically. Report both the old
in-sample scores and the new recovery scores so I can see exactly what
changes.

**One thing to carry into the decomposition.** The location-versus-shape split
below should be reported against `n_modes_visible` as well as against the
continuous modality measures. Stage 2a-2 found 94.9 percent of empirical
datasets have a single visible mode, which makes "KDE wins because of
multimodality" an assumption the data may not support. If the KDE's advantage
is concentrated in the visibly-unimodal majority, it is coming from skewness or
tail shape, and that is a different paper-level claim.

**Empirical: cross-validate, since no parent exists.** For each of the
empirical datasets with enough points, repeatedly split the data (carrying
weights) into fit and evaluation portions, fit all six UQ methods on the
fit portion, and compute W1 against the evaluation portion's weighted eCDF.
Report variability across splits and dependence on dataset size.

Two independent, non-circular lines of evidence. Report them side by side
throughout, and keep the empirical datasets in view for every headline
result rather than dropping them partway as the current analysis does.

**Decompose the error.** Split the total W1 of a uniform-weighted model
against a variable-weighted target into the model's fit error against its
own weighting scheme's eCDF, and the distance between the uniform and
variable eCDFs. Report separately. The second term is definitional and I
want it labeled as such.

**Report regret, not just win rate.** How often a method ranks first
understates the practical question, which is what it costs to use one
method everywhere. For each dataset compute regret: the W1 of a given
method minus the W1 of the best method for that dataset. Report mean,
median, and upper tail (90th and 95th percentile) of regret per method, on
both the recovery score and the cross-validated score.

---

### Stage 2d - decompose W1 between uniform and variable, and build the threshold

**Two housekeeping items first, both small.**

1. **Verify the ICE figure behind `MASS_ECC_CEILING`, or mark it unsourced.** The
   mass ceiling of 100 kgCO2e/kg is justified in the record by a highest
   building-product coefficient of about 13 kgCO2e/kg for primary aluminium,
   attributed to the ICE v3.0 database, and that figure came from a session's own
   knowledge rather than from any document in `refs/`. It will be printed in the
   paper. Either find a citable source and record the exact figure, edition and
   page, or state in `reports/MANUSCRIPT_discrepancies.md` that the value is
   unsourced and that the ceiling rests instead on the independent arithmetic that
   does not need a database: combusting pure carbon yields 3.67 kg CO2 per kg of
   carbon, so 100 kgCO2e per kg of delivered product would require burning about
   27 kg of pure carbon for every kilogram shipped. That second argument is
   sufficient on its own, so this is about what the paper cites, not about whether
   the ceiling is right.
2. **Update the handoff spec in `CLAUDE.md`.** It now says explicitly that the
   handoff's only reader has no access to this repository: no `CLAUDE.md`, no
   `reports/`, no tables, no figures, no `refs/`, no source. Decision and entry
   numbers are trailing citations rather than substance, numbers and figures must
   appear as text, and an instruction to consult a repository file has to be
   addressed to the next Claude Code session rather than to that reader. Copy the
   current wording from this prompt's handoff section into `CLAUDE.md` so later
   stages inherit it.

**Four constraints inherited from Stage 2c, and they are not optional.** Score on
a non-circular target, which is `src/recovery.py`: the replayed parent on the
synthetic arm, cross-validation on the empirical one. Post-stratify every headline
with `recovery.post_stratify` and report both allocations. State a win share
rather than a mean rank on the empirical arm, and make no size-banded claim below
about n = 100 without the relative-gap view beside it. Never compare the two
weighting schemes out of sample on the empirical arm, because the weights are an
exchangeable flat Dirichlet draw, so a uniform fit predicts a held-out half by
construction.

**The task, as the author specified it.** For a given dataset, sample the flat
Dirichlet the study already uses and ask what proportion of those possible
weightings differ from uniform by more than the threshold this stage calibrates.
That converts the weighting question from a property of one arbitrary realization
into a per-dataset statement a practitioner can act on: assuming uniform weights
has an X percent chance of changing which material ranks first. It retires
`w_v_uw_wasserstein`, which is a single realization of a quantity whose draw-to-
draw spread Stage 2a-3 measured at up to 1.02 per dataset.
`audits/weighting_risk.py` is a working probe at 147 datasets by 300 draws in
seven seconds, so this is hours of work rather than days.

**Use A_IQR as the instrument, not the probe's thresholded CDF distance.** It is
KL2's own measure, the area of the interquartile range of the ensemble of PDFs
produced by sampling Dirichlet weights, so using it keeps this paper consistent
with the author's published work, and it is one number per dataset in density
space. Quartiles are pointwise in x and the component PDFs use the guarded
Silverman bandwidth. Two details still have to be read off
`refs/1-s2.0-S0921344926002466-main.pdf` rather than reconstructed: how the area
is normalized, and how many draws. That file is in the repository and this
instruction is for you, not for the handoff's reader; put what you find in the
handoff as text.

**One result here runs opposite to everything else in the study, and it deserves
its own paragraph rather than a footnote.** A_IQR correlates with the coefficient
of variation at Spearman +0.693 and with log(n) at -0.569. Every other finding in
this project makes dataset size the mechanism; this is the one place dispersion
beats size. Report it as such rather than smoothing it into the size story.

Read `CLAUDE.md` at the repository root for project context before
starting. Then commit any uncommitted work on the current branch
with a clear message, then create and check out a new branch named
`stage-2d-threshold`. Do all of this stage's work there. Commit as you go, in
logical units, so I can bisect later if a number moves unexpectedly. Tell
me the branch name and the commit you branched from.

W1 is bounded below by the absolute difference in means of the two
distributions. So the headline predictor, W1 between the uniform-weighted
and variable-weighted versions of a dataset, may be substantially measuring
how far the mean moves under reweighting rather than any change in shape.

- Decompose that W1 into a location component (absolute difference in
  weighted means) and a residual shape component. Report the ratio across
  all datasets.
- If it is mostly location, say so plainly. That is a good outcome: the
  practitioner rule collapses to something computable in a spreadsheet with
  no distributional machinery.

Then build the decision rule.

- Because datasets are normalized to a mean of 1.0, current W1 values are
  already W1 divided by the mean. Make that explicit and define a named
  relative measure. Verify invariance by rerunning a subset un-normalized.
- Also compute a version normalized by a robust scale (weighted IQR or
  weighted standard deviation) and report which behaves more stably.
- Calibrate against consequences, not against itself. For every pair of UQ
  methods within every pLCA, I have the relative W1 between the two models
  and whether the downstream decision changed: whether the identity of the
  top-contributing dataset flipped, and whether the full rank ordering
  changed. Fit P(decision flips) as a function of relative W1, by logistic
  regression or an isotonic fit, and give me the relative W1 at which flip
  probability crosses 1%, 5% and 10%, with confidence intervals. That curve
  is the deliverable.

---

### Stage 2e - pLCA construction: group size, resampling, and common random numbers

**Order of work, because one item is a prerequisite for a number already
published.**

1. **The sweep over materials per pLCA, first.** Stage 2d's flip thresholds --
   the probability that the top-contributing material changes crossing 1 percent
   at a relative W1 of 0.0018, 5 percent at 0.011 and 10 percent at 0.025 -- are
   all conditional on four materials of equal material use intensity. Four
   near-exchangeable materials make a ranking as fragile as it can be made, so
   those numbers are upper bounds. Sweeping the group size moves every one of
   them, so do it before anything that depends on them.
1b. **Sweep material use intensity properly. This is the more consequential of
   the two axes.** Author decision, added 2026-09-17. It replaces the
   "run a variant where one or two materials dominate" bullet in the resampling
   section below, which was a hand-built scenario rather than a sweep.

   Why it matters more than the count. Every dataset is divided by its own
   unweighted mean, so every material's mean contribution is 1.00 by
   construction, which means MUI carries all of the between-material variation in
   contribution. With every MUI at 1.0 the four materials are exactly
   exchangeable, and that is why Stage 2d's flip probabilities are an upper bound
   rather than an estimate. Concentration is the only lever that moves them toward
   what a building produces.

   - **Sample on the simplex, not in absolute units.** The building total is
     arbitrary, so MUI matters only through each material's share of the total
     mean contribution. That makes the object a share vector, and the study
     already samples share vectors: draw the MUI shares from a symmetric Dirichlet
     over the group's materials and sweep the concentration. Large concentration
     reproduces the equal-MUI case, which must be verified to match the current
     results exactly; concentration of 1 is flat on the simplex; small
     concentration puts nearly everything on one material. One parameter spans
     exchangeable to total dominance, and it reuses machinery already in
     `src/`. Do not sweep a log-normal scale parameter; the scale is not
     identified.
   - **Add a handful of deterministic checkpoints alongside the random draws**,
     because they are reproducible and interpretable and they bracket the sweep:
     MUI ratios of 1:1:1:1, 2:1:1:1, 10:1:1:1 and 100:1:1:1. Report these as named
     cases.
   - **Report against an observable, not against the concentration parameter.**
     Dirichlet concentration means nothing to a reader and its relationship to
     dominance changes with the number of materials, which would confound the two
     sweeps. Plot every outcome against the **ratio of the largest to the
     second-largest mean contribution**, and report the largest material's share of
     the total beside it. Both are computable by a practitioner from their own
     quantity take-off in one line, and the top-two ratio is the quantity that
     decides whether a flip is even possible: if the leading material's mean
     contribution is twice the next, no plausible difference between fitted
     distributions reorders them.
   - **The deliverable is the ratio at which the flip probability crosses one
     percent.** That is the sentence the paper wants: how much a material has to
     lead by before the choice of UQ method cannot change which one leads. Report
     the whole curve too, with the equal-MUI case labeled at ratio 1.0 as the upper
     bound, so Stage 2d's published thresholds stay interpretable rather than
     being replaced.
   - **Run Stage 2d's continuous outputs across the same sweep**, not just the
     flip probability. The 18 percent median change in a material's estimated
     contribution was measured at equal MUI, and it may well be the number that
     survives concentration best, since it does not depend on a ranking. If it
     does, that is a reason to lead with it.
   - **Anchor it if you can, and say so if you cannot.** One or two real ratios
     would let the paper name an operating point rather than only a curve. Check
     whether `refs/` holds the Marsh et al. (in press) staircase paper, which has
     four real designs with real quantities, and compute the top-two contribution
     ratio for each. If the quantities are not there, report the curve alone and
     state that no anchor was available. Do not construct a plausible bill of
     quantities.
   - **State the scope limit.** MUI is deterministic within a run. Real quantity
     take-offs carry their own uncertainty, which in practice can exceed the ECC
     uncertainty this paper is about. That is a limitation to write down, not a
     sweep to add.

2. **A guard against the smoke environment overwriting a results table.** In
   Stage 2d a smoke run reached a commit and overwrote the main pLCA results
   table, which was caught and restored within the session but only because
   someone noticed. The rule against it is currently a sentence in a document.
   Make it a check: the notebook refuses to write its main table when the smoke
   variable is set, or a test asserts the group count in the run metadata. This
   is ten minutes and it protects the most expensive artifact in the project.
3. **Common random numbers, reusing what exists.** A tested implementation is in
   `src/flip.py`. Read it before writing another. Giving two methods the same
   uniform draws makes their comparison exactly paired, and it is a refinement
   rather than a repair, because every continuous output already separates the six
   methods far more than sampling variation does.
4. **Do not re-derive the flip threshold.** It is calibrated, recorded as a
   constant with its provenance, and notebook 3 already prints the recomputed
   crossings beside the stored ones on every run so drift is visible.

**Then the experiment below, which is the most valuable single one left in the
project.** Run the pLCA twice on the same common random numbers: once with each
method's fitted models, and once with the true parents, which
`corpus.rebuild_parents` now replays byte-identically for every dataset in the
corpus. Then report how far each method's ECI Rank #1 Frequency sits from the
truth.

That converts every number in Stage 2c from "how close is the fitted CDF to the
data" into "how wrong is the answer a practitioner gets", which is the question
the paper actually asks. It is also the only experiment that can show the method
differences do not matter at the decision level, and if that is what it shows,
that is the cleanest result this paper could report: a practitioner would be told
the choice is safe rather than told to optimize it. Report it either way, and do
not bury a null.

The same four constraints from Stage 2c apply here: non-circular target,
post-stratify every headline and report both allocations, win share rather than
mean rank on the empirical arm, and never compare weighting schemes out of sample
on the empirical arm.

Read `CLAUDE.md` at the repository root for project context before
starting. Then commit any uncommitted work on the current branch
with a clear message, then create and check out a new branch named
`stage-2e-plca`. Do all of this stage's work there. Commit as you go, in
logical units, so I can bisect later if a number moves unexpectedly. Tell
me the branch name and the commit you branched from.

**Use common random numbers across UQ methods.** Each UQ method's Monte
Carlo draws are currently independent, so part of the measured difference
between two methods is sampling noise rather than method effect. At n =
10,000 this is small in aggregate, but the rank-flip analysis in Stage 2d
lives on near-ties, which is exactly where it is not negligible.

Implement it as: draw one uniform variate per material per Monte Carlo
iteration, then push that same variate through each of the six models'
inverse CDFs. Stage 2b already put all six on a common support of (0, inf)
with truncated, renormalized inverse CDFs and replaced rejection sampling, so
the machinery this needs is in place; use it rather than rebuilding it. Variates must be independent *across materials* within an
iteration, or you induce rank correlation between materials that should not
exist, and identical *across UQ methods*, which is the whole point. This
follows the practice recommended for comparative probabilistic LCA by
Henriksson et al. (2015) and Heijungs (2021) and used by Marsh et al. (in
press). Report results with and without common random numbers so I can see
how much of the previously measured difference was noise.

**Sweep the number of materials and resample the groupings.** The 10,000
datasets are currently partitioned once into 2,500 disjoint groups of four
by a single shuffle. Replace with a resampling design:

- Sweep the number of materials per pLCA over n = 2, 3, 4, 6, 8, 12.
- For each n, draw many groupings with replacement rather than one disjoint
  partition, so results carry bootstrap confidence intervals.
- Report how the effect of UQ method selection scales with n. I expect it to
  dilute as n grows because each material's share of the total shrinks.
  Confirm or refute and quantify.
- Material use intensity is handled by the sweep in task 1b above, which
  replaces the hand-built "one or two materials dominate" variant this bullet
  used to specify. Cross the two sweeps rather than running them separately: the
  number of materials and how concentrated their contributions are interact,
  because adding a material raises flip risk when contributions are even and
  barely moves it when the new material is small.

Attach bootstrap confidence intervals to every headline percentage and
NRMSE the analysis reports. None currently have uncertainty attached, which
is awkward in a paper about uncertainty.

---

### Stage 2f - replace univariate rolling averages

Read `CLAUDE.md` at the repository root for project context before
starting. Then commit any uncommitted work on the current branch
with a clear message, then create and check out a new branch named
`stage-2f-multivariate`. Do all of this stage's work there. Commit as you go, in
logical units, so I can bisect later if a number moves unexpectedly. Tell
me the branch name and the commit you branched from.

The relationship between goodness-of-fit and dataset characteristics is
currently shown as rolling averages (250 points each side) of W1 against
about 10 statistical metrics across 18 panels. Problems: no uncertainty
band, no indication of data density so sparse regions produce artifacts,
and the metrics are correlated with each other so 18 marginal views
overstate how many independent effects exist.

**First, resolve the Shapiro-Wilk inconsistency**, because it may be the reason
the normality metrics behave oddly and it must be settled before deciding which
metrics survive. `shapiro_wilk_weighted` returns the true Shapiro-Wilk statistic
for uniform weights and a Shapiro-Francia statistic for non-uniform weights, so
`fit_norm_SW` and `fit_norm_SW_uw` are different statistics that the analysis
compares as though they were one. Four panels of the main figure and the
paragraphs discussing them rest on that comparison. Pick one estimator and use
it for both columns, or report them as the distinct statistics they are.
**The test this stage should apply, which Stage 2e made possible.** A
characteristic that predicts the distance between a fitted curve and its target,
but not the error in the ANSWER a probabilistic LCA gives, is not worth keeping.
Stage 2e ran the pLCA against the true distributions the synthetic data was drawn
from, so the per-material errors against truth are on disk and this is now a
direct test rather than an argument. Run the reduction twice, once against the
fit score and once against the downstream error, and report any characteristic
that survives one and not the other; those are the interesting ones.

**A reduction that ends with dataset size and little else is a result, not a
failure.** Size is the only mechanism anything in this project has found: Stage
2c put log(n) at an R2 of 0.616 and 0.395 with nothing else earning a place, and
Stage 2e found the methods fail on the same materials, correlating 0.892 to 0.970
within a weighting scheme, so family choice does not hedge. If the reduction
lands on size plus the weighting scheme, say so plainly rather than retaining
characteristics to reach a target count.

**Stage 2c already answered part of this stage's question, so start from its
answer rather than rediscovering it.** Regressing per-dataset
`log(W1_KDE / W1_lognormal)` on `log(n)` and then adding each characteristic in
turn, `log(n)` carries an R2 of 0.616 and 0.395 on the synthetic arm and nothing
else earns a place. Modality is **last of eleven**, with an incremental R2 of
0.00002 at p = 0.95 on the empirical arm. The strongest addition is
`w_v_uw_wasserstein` at 0.046 and 0.055, and it is a property of the weight vector
rather than of the data, so a practitioner cannot compute it before choosing
weights; it is Stage 2d's quantity. Your job is the full multivariate reduction
across all methods and both arms, which 2c deliberately did not do. If it
contradicts the targeted result, say so directly; if it confirms it, the
confirmation is worth stating because the two were computed differently.

**Dataset size is a confound in everything this stage models, and it has a fix
already built.** Stage 2b found the KDE's advantage is a large-dataset advantage:
its mean rank improves monotonically with n on both arms while the lognormal's
degrades, and the two arms only appear to disagree because the corpus holds 25
percent of datasets above n = 1,000 against the empirical arm's 7.4 percent.
Include n as a first-class predictor rather than one metric among nineteen,
report every aggregate both equal-allocation and post-stratified to the empirical
size mix, and check whether any metric that predicts which method wins is really
predicting n.

**One conceptual question this stage owns.** The tuning objective matches the
shape of the synthetic characteristic distribution to the empirical one, while
the study also needs to span that space with margin so conclusions generalize
past the categories EC3 happens to hold. Those two goals can pull apart, because
matching concentrates the corpus where real data is dense. Stage 2a-3 confirmed
they do *not* pull apart on the known coverage shortfall, where the corpus is
short of the empirical upper tail on both counts at once. Your multivariate model
is what shows whether they pull apart anywhere else: if the metrics that predict
which method wins are ones where the corpus is densest, the generalization claim
is narrower than the coverage figure suggests. Report that directly.

**There are now three modality measures, and they disagree by design.**
`n_modes_visible` counts local maxima of a Scott's-bandwidth KDE with a 5
percent prominence threshold; `crit_bw_1` is Silverman's critical bandwidth;
`modality_index` is the continuous index. On the empirical data they disagree
profoundly: 94.9 percent of datasets have exactly one visible mode while only
about half are unimodal by Silverman. Treat them as three candidate predictors
rather than three readings of one thing, and report which of them actually
predicts W1 and which method wins. That is the question Stage 2a-2 could not
answer and it belongs here. Note also that the Silverman share carries a few
points of estimator noise, reading 49.3 percent at 100 bootstrap replicates and
45.6 at 60, so quote `nboot` with any figure derived from it.

**Before the modeling: stratum 1 has undefined metrics by construction.**
Kurtosis is undefined in 24.8 percent of stratum 1, because it divides by
(n-3) and that stratum runs n = 3 to 9. Any complete-case model will drop those
datasets silently, leaving you modeling a corpus that excludes its smallest
datasets, which is precisely the regime where parametric families are expected
to beat KDE. Report how many datasets each model actually uses, per stratum,
and handle the missingness explicitly rather than by default.

**Decided: use Shapiro-Francia for both columns.** It is the only one of the
two with a weighted form, so it is the only choice that makes the
uniform-versus-variable comparison a comparison of one statistic under two
weightings, which is what those four panels claim to show. Quantify how much
the affected metrics move once the uniform column switches, and note that the
docstring claims equivalence for n at least 20 while the smallest stratum here
is n = 3 to 9, so the equivalence argument does not hold where it matters
most. Also fix `_royston_pvalue`, which applies the n >= 12
polynomials to the 4 <= n <= 11 branch; no current result depends on it because
only the statistic is kept, but leaving a known-wrong p-value in a public
deposit is not acceptable.

Then the main work:

- Use the correlation structure established in Stage 2a Part 6.
- Fit a multivariate model of W1, and separately of which UQ method wins, on
  all metrics at once. Use something interpretable alongside something
  flexible: a GAM or regularized regression plus gradient boosting with
  permutation importance and partial dependence. Rank metrics by independent
  predictive contribution.
- Tell me which small subset, I expect three to five, carries essentially
  all the signal, so I can cut the figure down.
- For the survivors, replace rolling averages with LOWESS or binned means
  plus bootstrap confidence bands, and include a density rug so sparse
  regions are visible.
- Keep the rolling-average version available for comparison so I can check
  the new version tells the same story.

Note for the manuscript: the count is settled and the manuscript may be
right. Stage 1 recovered the `metrics` definition from the commented-out
cell 62 that made NB2 unrunnable, and it resolves to exactly 19 metrics, all
with labels, in a 4-by-5 grid consistent with the committed figure's aspect
ratio. So the figure has 19 panels, not the 18 counted in Stage 0, covering
about 10 distinct metrics: eight with uniform and variable versions, plus
dataset size and the uniform-to-variable W1 with one panel each, plus one
more that the recovered list should identify. Confirm the exact composition
from `src/fitting.py` and the recovered list, and tell me whether the
manuscript's "19 statistical metrics" is now simply correct.

---

### Stage 2g - sensitivity of the headline decision metric

**A third independent argument for demoting the ranking metrics arrived from
Stage 2f, and this one is about predictability rather than fragility.** The error
in a material's chance of leading is only 9 percent predictable from its own
dataset's characteristics, against 62 to 66 percent for the error in its
estimated contribution, because a rank-1 frequency is a property of the group a
material is placed in and not of the material's own data. Taken with the
fragility argument from Stage 2d and the noise argument from Stage 2e, that is
three separate reasons to lead with magnitudes.

**Stage 2e made a specific recommendation for the metric set and it should be
the starting point.** Report the five statements a probabilistic LCA actually
makes, in this order, because the order is the results section:

1. **The design comparison, which leads, but as a threshold rather than as a
   null. The unqualified null is withdrawn.** It said the choice of method changes
   the stated probability that a substitution is an improvement by at most 0.015,
   with every method within 0.026 of the truth. Those are errors in the AVERAGE
   probability over many comparisons, because that row and the four
   reduction-strategy rows averaged signed errors over groups before taking the
   absolute value, so the errors cancelled. Per comparison the six methods span a
   median of 0.165 at a claimed 5 percent saving and disagree about which design
   is better on 28 percent of comparisons: 64 percent at a claimed 2 percent
   saving, 28 at 5, 5.9 at 10, and 0.04 at 20.

   **State it as a threshold, which the data supports and which is a better result
   than the null was.** A design comparison is method-independent when the claimed
   saving is large enough and method-dependent near a tie. That is the same
   structure as the safe-lead rule below, where a material must lead the next by
   about a factor of two, and as the flip thresholds. One mechanism, three
   appearances: near a tie, fragile; clear of a tie, safe. A paper that says that
   about three different claims is making a general point about probabilistic LCA
   rather than issuing three separate caveats.

   **Keep the averaged number, labelled for what it is.** The error in the average
   probability across many comparisons is the right quantity for a building-stock
   or portfolio study and the wrong one for a single design decision. Both belong
   in the paper; neither should be quoted as the other.

   **The number a practitioner needs is how often a method names the wrong
   design**, measured against the truth run on the same draws over 2,500 pairs.
   By claimed saving, the range across the four non-normal methods: 27 to 33
   percent at 0, 25 to 29 at 2, 16 to 17 at 5, 3.8 to 5.6 at 10, and 0.0 at 20.

   Two readings of that, and the paper needs both. At a claimed 5 percent saving
   every method names the wrong design about one time in six and the six sit
   within 6 points of each other, so the choice of method barely matters and the
   error rate is high anyway. That is the clearest instance in the study of the
   distinction between what the choice costs and how good the answer is at all,
   and it should be used to make that point concrete.

   **And the normal fails at the decision level, not merely at the fit level,
   which is the strongest single argument against it in the project.** At a
   claimed 0 percent saving `Normal, Uniform` names the wrong design 65.8 percent
   of the time, worse than a coin toss, and `Normal, Sampled` 46.9 percent. That
   is not noise: the normal is systematically optimistic, so when the truth sits
   near a coin flip a systematic lean puts it on the wrong side more often than
   chance would. By a claimed 10 percent saving it is indistinguishable from the
   others. Same mechanism as the bias that accumulates across a building's
   materials, now showing up in a single binary decision.

   **One caveat belongs with the top rows.** At a claimed 0 to 2 percent saving,
   12 percent of pairs have a true probability within 0.02 of a coin flip, where
   naming a side is meaningless for anyone. Report the wrong-design rate both over
   all pairs and excluding those, so the headline is not carried by comparisons no
   method could get right.

   **The rule to print, measured on a dense sweep bracketing every crossing within
   two points, with a cluster bootstrap over design pairs.** A substitution claimed
   to save more than about **10 percent** of the building is called correctly by
   any of the six methods 95 times in 100, and more than about **15 percent**, 99
   times in 100. Below about 5 percent no method is reliable and choosing between
   them does not help. The crossings: a method names the wrong design 5 percent of
   the time at 10.15 percent claimed saving [9.57, 10.73] and 1 percent at 14.57
   [13.46, 15.21]; two methods disagree with each other 5 percent of the time at
   11.10 [10.42, 11.68] and 1 percent at 15.20 [14.12, 15.81]. Quote these to two
   significant figures, per the standing rule on fitted constants, so 10 and 15 are
   what the text says. The sweep agrees with the production run where they overlap,
   18.01 percent against 19.76 at a claimed 5 percent saving and 4.68 against 5.20
   at 10.

   **This is the paper's second threshold rule and it should be presented beside
   the first.** Attribution is safe once a material leads the next by about a
   factor of two; a design comparison is safe once the claimed saving exceeds about
   a tenth of the building. Same mechanism, two decisions, two numbers a designer
   can check against their own model.
2. **The safe-lead rule.** The probability that the choice of method changes which
   material leads crosses 1 percent at a top-two contribution ratio of **2.1**,
   not 2.13; see the two-significant-figure rule below. The one real element
   available, the
   Concrete-Precast staircase in Marsh et al. (in press), sits at 1.02, so real
   designs do land in the fragile zone.
3. **The building total and the budget statement**, where every method understates
   the 90th percentile and two methods disagree about meeting a 90 percent budget
   by 91.3 against 86.8.
4. **The specification result, which is where the choice costs the most.** Against
   a true mean saving of 5.39 percent of the building from capping a material at
   the 75th percentile of declarations held, the truth gives a 23.2 percent chance
   of achieving at least 5 percent and the normal says 30.5.
5. **The contrast that explains the stage**: a quantity reduction is
   method-independent to four decimal places because it is a deterministic
   fraction of a material's own contribution, while a specification cap acts
   entirely through the upper tail, which is exactly what the methods disagree
   about.

**Add the uncertainty index, which is the steadiest output measured and appears
in no table, figure or section of this study.** Its NRMSE between methods is
0.503 against 1.042 for a material's rank-1 frequency, and it is the fourth
reduction strategy under another name: three strategies reduce the expected
impact, use less, specify better, substitute, and this one reduces the variance
of the answer, which is what buying a supplier-specific declaration actually
gets you. Evaluate it against the parent-truth comparison like any other
candidate, but the case for it is already strong.

**Finish the `(1 - capecc)` divisor.** Stage 2e resolved it partly by replacing
the relative cap with an absolute one, which retired the divisor because it
scaled a count by 1/0.25 and that was exact only while the old cap bound in
exactly 25 percent of iterations for every material by construction.
`capecc_rank_1` moved from a mean of 0.684 to 0.193 as a result, which is a
change of definition rather than of substance. Confirm the new definition is the
one the paper wants and that nothing downstream still assumes the old one.

**The decisive comparison arrives here from Stage 2e.** Once the pLCA has been
run against the true parents, this stage's question changes from "is this metric
sensitive" to "does this metric recover the right answer". Judge every candidate
headline metric, the current ECI Rank #1 Frequency and any companion you propose,
by how closely it reproduces the parent-truth result. Also fix notebook 3 cell 45,
which explains pLCA outcomes using the in-sample fit score, a target Stage 2c
retired.

**Two things arrive here from Stage 2b.** First, the empirical headline should be
a win share rather than a mean rank, because over five weight realizations the top
two methods are within the draw noise of each other on mean rank while the win
share separates them cleanly in every realization. Build the win-share view
alongside whatever companions this stage proposes. Second, this stage owns the
downstream end of a question Stage 2c owns from the front: a goodness-of-fit
statistic between CDFs is nearly blind to tail mass and the pLCA is not, because
it samples. A model that scores well while carrying a thin enormous tail will
dominate any Monte Carlo it enters. Check any recommended metric against that
failure mode rather than assuming it immune.

Read `CLAUDE.md` at the repository root for project context before
starting. Then commit any uncommitted work on the current branch
with a clear message, then create and check out a new branch named
`stage-2g-metric`. Do all of this stage's work there. Commit as you go, in
logical units, so I can bisect later if a number moves unexpectedly. Tell
me the branch name and the commit you branched from.

The main downstream metric, ECI Rank #1 Frequency, sits near 25 percent by
construction with four overlapping distributions and equal MUIs, and may be
fragile to small perturbations. Add magnitude-based companions: each
dataset's mean share of the total, and its share at the 95th percentile of
the total. Check whether conclusions hold under those as well as under the
rank metric. Marsh et al. (in press) find the same instability in a real
staircase design, where the top-contributing product changes with the
uncertainty characterisation scenario, so this is worth reporting carefully
rather than defensively.

Also resolve the `(1-capecc)` divisor applied to the cap rank frequencies,
which carries the inline comment that the percentages look off because not
all reduction strategies apply in all scenarios. Work out what the correct
normalization is for that quantity, implement it, and report how the case
study numbers change. An ad hoc divisor with a comment conceding the
numbers look wrong is the kind of thing a reviewer finds in deposited code.

---

### Stage 2h - robustness sweeps

Read `CLAUDE.md` at the repository root for project context before
starting. Then commit any uncommitted work on the current branch
with a clear message, then create and check out a new branch named
`stage-2h-robustness`. Do all of this stage's work there. Commit as you go, in
logical units, so I can bisect later if a number moves unexpectedly. Tell
me the branch name and the commit you branched from.

Each of these closes a "you only tested one variant" objection. Run them as
sweeps with tabulated results, not one-off checks.

**Two items that were in this section and are now closed, so that nothing here
reopens them.** The bandwidth is settled: the study uses Silverman's rule with a
guard, adopted in Stage 2c after Stage 2b diagnosed and fixed its small-sample
failure, and it beat Scott on held-out likelihood. It is a sensitivity in this
stage, not an open choice, and the item below says so. The lognormal offset is
retired: the study fits a three-parameter lognormal by profile likelihood, and
what replaced the offset is `PROFILE_DELTA_LO_FRAC`, swept below. Do not sweep an
additive offset; there is none.
- **Multiple weight realizations, and this one could move a headline.** Stage
  2a-3 found that redrawing the Dirichlet weights of the same datasets from the
  same distribution, changing nothing else, moves `w_v_uw_wasserstein` by up to
  1.02 in absolute terms per dataset, excess kurtosis by 365.8 and the
  coefficient of variation by 4.09, while every unweighted column stays
  bit-identical. The arm-level distribution is far more stable and that is what
  the study rests on, but the manuscript does not currently distinguish a
  per-dataset weighted statistic from the distribution of them. Quantify both,
  and say which claims in the paper depend on which.
- **Before the sweeps: make all sixteen scorecard rows the same quantity. The
  author has decided to take the correction.** Five of the sixteen claims, the
  four reduction-strategy rows and the design comparison, read summary tables
  whose errors were averaged over groups before the absolute value was taken, so
  signed errors cancel and those five flatter themselves. Per claim's true level,
  best method: how often a cap binds reads 0.48 against 33.01 per group; what
  using 25 percent less saves reads 0.00 against 10.02; the design comparison
  reads 0.81 against 11.95.

  Recompute all five per unit so every cell of the figure is what its caption
  says it is. The figure's headline moves from "right to 0.8 percent" to "right to
  12.0 percent" and that is the correct number. Report the before and after for
  all five rows, and mark clearly which published statements change, because the
  design comparison is the claim the results section currently opens with.

  **This is a change of definition, not of substance, and the averaged number is
  still worth reporting** as what it actually is: the error in the AVERAGE
  probability across many comparisons, which is the right quantity for a
  portfolio or a stock model and the wrong one for a single design decision.
  Report both, labelled.

- **A standing rule for every fitted constant, settled at the close of Stage 2g
  and applying to anything this stage fits.** The published crossings were checked
  and none is an extrapolation: the flip thresholds are bracketed by binned
  observations either side over 22,500 calibration rows, and the safe-lead ratios
  over 72,000 four-material comparisons. Nothing needs re-deriving. But the
  intervals on them were bootstrap intervals on a fitted logistic's parameters,
  which say how well the data pin down that model and nothing about whether a
  logistic is the right shape, and an isotonic fit falls outside the published
  interval on four of six. So: **fit every crossing both ways, parametric and
  monotone-nonparametric, and report both values with the bootstrap interval, at
  full precision, in the results tables and the supplement.** Nothing is thrown
  away: a reader checking the work, or using a constant downstream, needs the
  unrounded value and both fits.

  **The rounding applies only where a constant is stated as a rule in prose or in
  a figure annotation. There, round at the first digit where the two fits
  disagree.** A printed digit is read as measured, and on these constants the
  third digit is determined by the choice of fitting family rather than by the
  data, so printing it claims a precision the study does not have and implies a
  cliff where the underlying curve is smooth. It would also fail to reproduce: a
  reader refitting with a different family gets a different third digit and
  reasonably concludes something is wrong.

  On the current constants that gives 2.1 for the 1 percent safe lead, where
  logistic and isotonic give 2.13 and 2.22, and it exposes the 10 percent safe
  lead as 1.5 against 1.4, which is an honest spread rather than a rounding. It is
  not a fixed count: where the two fits agree to four figures, the prose may print
  four. Apply the same treatment to any crossing this stage produces.

- **The two arms weight their data by different rules, and fixing that comes
  second.** Stage 2f found it while looking at something else. The real material
  categories draw market shares from a flat draw over their individual
  declarations, so shares are independent of the carbon coefficients. The
  synthetic datasets attach shares to the humps of the distribution and split
  within each hump, so shares are correlated with the coefficients. That is a
  structural inconsistency on the exact quantity the paper is built on.

  It is measurable and it behaves as independence predicts. The median distance
  between the equal-weighted and market-share-weighted versions of the same data
  declines with size at -0.397 on the real categories against -0.167 on the
  synthetic, and above a thousand declarations the synthetic arm shows ten times
  the effect. Reweighting the synthetic arm's own values by the real arm's rule
  reproduces the real arm's behavior, which proves the gap is the rule and not
  the data. The two agree at 10 to 99 declarations, where 78 of the 147 real
  categories sit, which is why nothing caught it for six stages.

  **The direction of the fix is settled: port the synthetic arm's method to the
  empirical arm, not the reverse, and not a compromise between them.** The
  synthetic method attaches a share to each mode and splits it within the mode. The
  empirical method draws a flat Dirichlet over every individual declaration, which
  asserts that market share is unrelated to carbon intensity and, because
  independent weights wash out as a category grows, guarantees that weighting stops
  mattering once a category is large. On a category holding hundreds of
  declarations that is not a neutral default, it is a strong and false claim about
  markets, and it is the reason the real arm's weighting effect decays at -0.397
  against the synthetic -0.167. The synthetic method is the better model of a real
  market and the empirical arm should be brought to it.

  **The obstacle is that real data does not come with mode labels, so the port
  needs a mode proxy that is estimable at every dataset size.** Do not use a fitted
  mixture model: it cannot be estimated at three to nine declarations, mode counts
  on real data swing from 95 to 68 percent unimodal on one smoothing choice, and it
  would put a modeling decision inside the paper's central quantity. Cutting the
  sorted values into contiguous blocks is the proxy to try first, because it
  reproduces the structure that matters -- a share attached to a group of adjacent
  coefficients, split within the group -- with no fitting and no failure mode at
  small n. It is also the same block structure this stage was already told to
  sweep, so the sweep and the fix are one piece of work.

  Validate the proxy rather than assuming it: apply it to the SYNTHETIC arm, where
  the true mode labels are known, and report how closely block-derived weights
  reproduce the weighting effect that the true labels give. That is a direct
  measurement of what the proxy costs, and it is available only because the
  synthetic arm has both.

  **Declining to choose a clustering strength is itself a choice**, and the wrong
  one: assuming no clustering asserts that market share is unrelated to carbon
  intensity, and published production volumes say otherwise, with 63.75 percent
  of world steel on the higher-carbon route. Sweep the strength, anchor it where
  published volumes allow, and report the headline weighting claims across the
  range rather than at one setting.

  **Two consequences to handle explicitly.** Stage 2f's finding that equal
  weighting beats market-share weighting below about a hundred declarations rests
  on this rule and may not survive it; do not let any recommendation about
  weighting at small sizes stand until this reports. And the generator's tuning
  objective reads market-weighted columns, so changing the rule may move the
  calibration. **Measure that against the seed-to-seed noise and report it. Do not
  regenerate.** Reopening generation is an author decision, and the standing rule
  is that a finding of this kind becomes a reported limitation or a sensitivity
  rather than a rebuild.
- **Revisit the corpus's joint modality-and-dispersion structure, together with
  the weight model.** The author has authorized regenerating the corpus if the two
  together make it work; see the standing section on reopening generation for how
  to go about it.

  The problem: the corpus reproduces modality and dispersion each on its own but
  not jointly. Conditional on being dispersed, a real category has more than one
  visible hump 44.7 percent of the time against a synthetic 9.4, and categories
  that are both are 16.2 percent of real against 1.9 percent of synthetic, with
  the correlation opposite in sign on all six characteristics measured.

  The reason is mechanical: the generator makes a dataset widely spread by
  sliding the whole distribution toward zero rather than by moving a second hump
  away, correlation -0.654 against -0.017, and the smallest admissible slide grows
  with the width, so separating humps actively costs spread. A fix that sets hump
  spacing to produce the spread directly was written, tested and defaulted off; it
  is `genconfig.separation_dispersion_frac`, with `genconfig.shoulder_frac`
  alongside it, both added in Stage 2g and both currently 0.0. It
  takes the both-at-once share from 1.3 percent to 12.7, and it makes the corpus
  overstate the paper's headline weighting effect by a factor of 2.7, with the
  typical dataset's weighting effect going from 0.1035 against a real 0.0929 to
  0.2513.

  **The reason to try again here rather than accept it as a limitation is that
  the two are coupled.** That 2.7 factor is measured against the CURRENT weight
  model, which is the one this stage is replacing, and the current model is known
  to be wrong: the two arms draw shares by different rules. A weight model that
  correlates shares with coefficients on both arms changes what the weighting
  effect is, so it may change the trade. Settle the weight rule first, then
  re-measure the hump-spacing fix against it.

  **The criterion is the paper's headline weighting effect, not the joint
  structure.** Adopt the generation change only if, under the settled weight
  model, it closes the joint-structure gap without making the corpus overstate the
  weighting effect relative to the real arm. If the trade persists, report both
  measurements and leave generation alone; thirty-six configurations were searched
  before and the same trade has appeared on three levers across three stages, so a
  failure here is informative rather than a dead end. Either way, report what you
  measured. If you do regenerate, everything downstream of the corpus re-runs, so
  say so before starting rather than after.

- **The pedigree matrix, added by the author at the close of Stage 2e, and the
  third most important item in this stage.** Every method this study compares is
  data-driven. The probabilistic LCA methods in general use, above all the
  pedigree matrix, are formulaic expert judgment applied where data is absent.
  They cannot be compared like for like, and this stage should not claim they
  can. But Stage 2e's yardstick, error against the true distributions the data was
  drawn from, does not care how a model was built, so a judgment-driven model can
  be placed on the same axis without asserting the two approaches are comparable
  in kind.

  **Sweep two dimensions, not one: the spread AND the location.** The geometric
  standard deviation is the obvious axis and it is not sufficient on its own,
  because a pedigree-matrix model is a spread applied around a point estimate the
  practitioner already holds, and there is no reason that point sits where the
  market-weighted mean of the category sits. A practitioner working without data
  typically centers on a single declaration they happened to obtain, a generic
  database value, or an industry average, and each of those is offset from the
  category's true mean by an amount nobody can see.

  That offset is likely to matter more than the spread, for the reason this
  project has already established twice: bias adds across a building's materials
  while random error cancels, and bias is what puts a method on the wrong side of
  a near-tie decision. A judgment-driven model with a well-chosen spread and a
  displaced center will fail in exactly the way the normal fit does. Sweeping only
  the spread would miss that and would flatter the pedigree approach.

  **The realistic location model needs no free parameter and should be the primary
  case.** A practitioner with no data has one declaration, so draw the center as a
  single random declaration from the category and let the offset distribution fall
  out of the data rather than being chosen. Sweep a deliberate offset as a fraction
  of the true mean alongside it, so the sensitivity is mapped rather than only
  sampled, and report the two dimensions jointly rather than one at a time.

  **Sweep the geometric standard deviation across the range the matrix produces
  rather than choosing pedigree scores.** The scores describe a data-collection
  context that a generated dataset does not have, so selecting them would be
  inventing a provenance. The question is: at what spread does a judgment-driven
  model start to give different answers from a data-driven one? Report it against
  the same outputs as the five statements, above all the design comparison, since
  a null there would say the practice most readers use is also safe for the
  decision they make.

  This is what connects the paper to the practice its audience actually uses, so
  give it room rather than treating it as one sweep among eight.
- **The Dirichlet concentration sweep must vary the BLOCK STRUCTURE, not only the
  concentration parameter.** Stage 2a-3 established that concentration alone is
  captured by effective sample size while coherence is not, and that clustering
  share on products with adjacent coefficients gives 1.5 to 3.1 times the
  separation at matched effective sample size. A sweep over the parameter alone
  will therefore miss the effect that makes every weighting number in the paper a
  lower bound.
- **An industry-average EPD is a direct estimate of the market-weighted mean, and
  the empirical extract may already contain some.** If EC3 flags industry-wide or
  industry-average declarations, compare that value against the uniform mean of
  the individual declarations in the same category. That is a real-world check on
  the study's central weighting claim, using data already on disk, and nothing
  else in the project can test it directly. Gate it on the data supporting it: if
  the flag is absent or the coverage is too thin, say so and drop it rather than
  inferring which declarations are industry averages from their names.
- **The bandwidth, which is now a sensitivity rather than an open choice.**
  Stage 2c adopts `silverman_guarded` with `SILVERMAN_MIN_NEFF = 20.0`, on Kish
  effective sample size, swapping the scale estimate only while the coefficient
  stays at 0.9. Sweep
  around it: Scott, pure Silverman, the guarded rule at several minimum effective
  sample sizes, and cross-validated bandwidth. Judge on held-out likelihood and on
  the parent-based score from 2c, never on in-sample W1, which falls monotonically
  as the bandwidth shrinks and therefore cannot arbitrate. Report what the
  headline does under Scott specifically, because that is the configuration the
  manuscript was written against.
- **Two constraints on which families may enter the main comparison, and one on
  how any winner is reported.** A family belongs in the main comparison only if it
  can use the data. A uniform fitted to n declarations is the smallest and the
  largest of them and nothing else, so putting it beside a kernel estimate would
  manufacture a win for this paper's own recommendation. **Uniform and triangular
  go in the judgment arm with the pedigree matrix**, where the question is how far
  a model built without data sits from one built with it; gamma and Weibull are
  data-driven and belong in the main comparison. And every "method X is best"
  claim in this project is a claim about one metric: on two of the seven metrics
  Stage 2g measured, no method separates from the runner-up at all, and the
  ordering of the six is negatively correlated with a material's chance of being
  the largest contributor on three of six companions. **Name the metric a winner won on and show
  the interval.** A leader whose interval overlaps the runner-up is not a leader.
- **An upper truncation of each fitted model, which would remove the
  thin-far-tail failure mode outright.** Every model is already truncated below at
  zero, because a negative emission coefficient is inadmissible and that bound
  needs no argument. An upper bound has no equally external anchor, since every
  dataset here is rescaled to an average of 1.0, so choosing one is a modeling
  decision with numbers attached, which is why Stage 2g stated it rather than
  implementing it. Sweep it; report what it costs and what it buys.
- **The three-parameter lognormal's bounds, of which `PROFILE_DELTA_LO_FRAC` at
  0.25 is the one that binds.** The others are `PROFILE_DELTA_HI_FRAC` at 1000.0,
  which is effectively non-binding, and `PROFILE_GRID_POINTS` at 400. Sweep the
  first, and check the other two are as non-binding as they look rather than
  assuming it. It is the threshold guard on the profile-likelihood
  lognormal, it binds on about half the empirical arm, and Stage 2b showed the
  fitted model's usability is far more sensitive to it than W1 is: at a guard of
  0.01 the largest fitted standard deviation reached 532 on data whose own is near
  0.6, with W1 barely moving. **Report the maximum fitted model standard deviation
  alongside W1 at every point in this sweep**, or the sweep will not see the thing
  that matters. **The tail correction in the scoring criterion must stay in force
  throughout.** Without it the criterion returns the same number to seven
  significant figures whether the misplaced mass sits at a hundred times the
  dataset mean or at a thousand, because it integrates over a grid ending at 9.9
  times the mean. That guard is the only thing making the level metrics Stage 2g
  recommends safe to report, so sweeping it with the correction off would be
  measuring nothing. One thing worth knowing first: every one of the worst runaway
  tails in the study, at 73.7, 50.1 and 40.6 times the data's own spread, is an
  EQUAL-WEIGHTED fit to a small dataset, and each one's market-share-weighted twin
  tops out between 1.0 and 5.0. That may make this and the mode-share
  concentration sweep the same phenomenon.
- **`min_q1_over_iqr`, and a measured partial win already in hand.** Stage 2a-3
  measured moving it from 0.5 to 0.05: the coefficient-of-variation distribution
  distance improves from 0.273 to 0.199 and visible-mode total variation from
  0.007 to 0.005, with the objective flat. It did not adopt it, because the
  parameter is yours and adopting it meant another regeneration for a gain inside
  noise. Sweep it, report whether it narrows the coverage shortfall in entry 34
  as a side effect, and treat any improvement as a reported sensitivity rather
  than grounds to rebuild the corpus.
- **`min_mode_sd_frac`, currently 0.15.** A judgment with no empirical anchor:
  the comparable empirical quantity needs a fitted mixture and 39.7 percent of
  empirical datasets sit exactly on `gmm_em_1d`'s variance floor, so there is
  nothing reliable to calibrate against. Sweep it and report what the corpus and
  the headline results do across the range.
- **`trunc_iqr_mult`.** Note before sweeping that the latent bug in
  `MixtureParent.truncated_moments`, fixed in Stage 2a-2 section 4.6, would have
  bitten this sweep specifically. Confirm the fix holds across the range you
  sweep.
- **Six-or-more-mode datasets.** 5.6 percent of the corpus against an empirical
  0.7 percent. Report whether it matters for any headline result before
  proposing a generation change, since generation is closed.
- **Two empirical-population sensitivities, both against the same primary.** The
  primary analysis is EPD-level uniform weighting over the split categories from
  Stage 2a-3. Report as sensitivities: (a) the deduplicated variant, unique
  (manufacturer, product); and (b) the unsplit arm, meaning the original EC3
  categories before Stage 2a-3's splits. The second one matters most: it is the
  evidence that splitting did not manufacture the headline result, so report it
  against every headline figure rather than only the aggregate.
- **More parametric families.** Add gamma and Weibull as additional
  estimation methods. Both are natively on (0, inf), so they need no
  truncation under the support decision in Stage 2b. Cheap to fit, and they
  blunt the objection that only two families were tested. Supplement if they
  crowd the main results.
- **Weight concentration and the prior-versus-truth question.** Variable
  weights are drawn from a Dirichlet and a single draw is treated as ground
  truth. That is defensible: every product behind an EPD does have a real
  discrete market share, and a flat Dirichlet is the maximum-entropy prior
  over unknown shares. But real market share is not uniform on the simplex;
  it is heavy-tailed. In the KL2 steel example China alone is 54 percent of
  global production, and in Marsh et al. (2025) Rest-of-World BOF is 63.75
  percent while Austrian EAF is 0.03 percent. So the flat Dirichlet is
  probably less concentrated than reality and the measured weighting effect
  is likely a lower bound. Sweep the concentration parameter from strongly
  concentrated to near uniform and report how the effect scales. I want to
  state in the paper that the reported effect is conservative, with
  evidence.
- **Weight realization.** Draw multiple independent weight vectors per
  dataset and report how much of the observed effect is specific to one
  realization.
- **Mode-to-point weight coupling.** Sweep the coupling parameter introduced
  in Stage 2a Part 3.

---

### Stage 2i - real building anchor (decided: not running)

Read `CLAUDE.md` at the repository root for project context before
starting. Then commit any uncommitted work on the current branch
with a clear message, then create and check out a new branch named
`stage-2i-anchor`. Do all of this stage's work there. Commit as you go, in
logical units, so I can bisect later if a number moves unexpectedly. Tell
me the branch name and the commit you branched from.

**Decided before Stage 2a: this stage does not run.** Marsh et al. (in press)
already demonstrate, on four real staircase designs with real quantities, that
the identity of the top-contributing product changes with the uncertainty
characterisation scenario. Citing that serves the "is this realistic"
objection better than running our own case, which would be validation of a
general result rather than a contribution, and it avoids a dependency on an
external quantities dataset. The manuscript makes the argument by citation.

The specification below is retained only in case that decision is revisited: take real material quantities from the harmonized
North American building LCA dataset of Benke et al. (2025), or DEQO or SE
2050, attach the corresponding empirical ECC datasets, and run the pLCA
under all six UQ methods. Report whether the UQ method changes which
material is the top contributor, and whether the relative W1 threshold from
Stage 2d predicts that outcome. Keep it small: one table, at most one
figure. It is validation supporting a general result, not a case study.

---

## Stage 2j - let the method vary by material

*Proposed at the close of Stage 2h and confirmed by the author. Runs before
Stage 3, because it may add a row to the scorecard figure. Rewritten
2026-09-25 by the Stage 2h window, which owns this file.*

Read `CLAUDE.md` at the repository root for project context before starting.
Read `reports/START_HERE.md` as well: the chat window that used to review each
stage is retired and this window owns the prompt file. Then
commit any uncommitted work on the current branch with a clear message, create
and check out `stage-2j-per-material`, and do all of this stage's work there.
Commit in logical units so a number change can be bisected. Report the branch
name and the commit you branched from.

**THE QUESTION.** Every simulated building in this study fits ONE method to all
four of its materials, so a material's own goodness-of-fit advantage is averaged
against three neighbours drawn at random from the whole corpus, most of them
below the size where that advantage exists. Stage 2h measured the consequence:
the kernel estimate overtakes the three-parameter lognormal on FIT at about 81
declarations and does not pull clear on the downstream CLAIMS until about ten
times that. Letting the method vary BY MATERIAL should recover much of the gap,
and unlike every other policy this study evaluates it is one a practitioner can
follow one material at a time, without knowing anything about the other three.

**THE RULE IS ONE NUMBER AND THE STUDY ALREADY HAS IT. DO NOT REDISCOVER IT.**

    at or above 81 declarations   kernel estimate, market weights
    below 81 declarations         three-parameter lognormal, uniform weights

81 is the practitioner threshold, and it was calibrated on the superseded corpus
and came back UNCHANGED on the current one, with the band of equally good
choices at 68 to 106. Both the family and the weighting switch at the same line,
because Stage 2h found both orderings invert at about 100 declarations.

**IT MUST NOT INVENT A SECOND SELECTOR.** Every other characteristic was tested
across two stages and none yields a usable threshold; modality as a selector is
measurably WORSE than not selecting at all. A rule with two numbers in it is not
the deliverable and would not survive review.

**THE IMPLEMENTATION IS CHEAP AND THE STAGE SHOULD NOT SPEND ITS TIME
ELSEWHERE.** `plca.run_group` and every function around it index
`models[dataset][method]`, so the mixed policy is a NEW KEY on each dataset's
existing dict pointing at whichever of the six already-fitted models the rule
selects. **Nothing is refitted.** Add the key, add its name to `methods`, and
the existing pLCA, truth run and scorecard machinery carry it through. Expect
roughly one sixth more run time than the six-method run, not a second run.

**WHAT THIS STAGE OWES.**

1. The run against the TRUE distributions, 2,500 groups, scored on the same
   sixteen claims as the scorecard, with all six fixed-method policies as
   controls in the same run and on the same random draws.
2. The result reported in the PER-UNIT form as primary, with the per-portfolio
   form beside it for the five rows where the two differ. Say which is which
   every time; for those five they differ by about a factor of twenty.
3. A plain statement of whether the mixed policy beats the best fixed policy,
   and by how much, against the noise of the comparison rather than against
   zero.
4. The same question asked of the FIT as well as the claims, because Stage 2h
   established that a fit advantage is heavily attenuated by the time it reaches
   a claim and this stage is the one that can say whether per-material selection
   is what recovers it.

**A NULL IS A RESULT HERE AND MUST BE REPORTED AS ONE.** If the mixed policy
does not beat the best fixed policy, that says the study's single-threshold rule
already captures what there is to capture, which is a cleaner recommendation
than a mixed policy and is worth stating plainly. Stage 2h's own strongest
result was a null of exactly this kind.

**WHAT MIGHT MAKE IT A NULL, so the stage recognizes it rather than chasing
it.** A probabilistic LCA claim belongs to the GROUP of four materials, and the
group's worst-fitted member sets much of the group's error. Improving one
material of four cannot improve the group more than that member's share of it.
Measure the group-composition effect directly -- for instance by splitting the
result by the SMALLEST dataset in each group, which Stage 2h found moves the
kernel estimate's advantage by a factor of nearly two -- so that a small overall
gain is attributable rather than merely disappointing.

**FOUR CONSTRAINTS CARRIED FROM STAGE 2h, ALL OF WHICH IT LEARNED THE HARD
WAY.**

- **Judge any calibration movement against the weight-draw noise of 0.006 to
  0.015**, not the generator seed noise of 0.0066. The weight draw is the larger
  of the two and six stages quoted the smaller one without knowing.
- **An absolute distance is not a quality score.** Every goodness-of-fit number
  in this study rose 16 to 45 percent when the corpus was made more dispersed,
  and the fits did not get worse. If this stage compares against any stored
  absolute band, re-measure the band first.
- **Use "market weights", "uniform weights" and "known market shares".**
  "Variable", "sampled market shares" and "Dirichlet shares" are retired and
  must not appear in an axis label, legend, panel title, column name or
  filename. The stored `method` values keep "Uniform" and "Variable" because
  they are the join key for every table and fixture.
- **Every audit table under `outputs/tables/audits/` dated 2026-09-24 was
  computed on the SUPERSEDED corpus and the pre-port weight rule.** Do not
  quote one. If this stage needs a number from the judgment arm, the
  upper-truncation sweep, the bandwidth sweep or the certification credit,
  re-run the script first -- each takes 8 to 35 minutes.

  **REFRESHING THEM IS NOT THIS STAGE'S JOB AND IS NOT ANY STAGE'S JOB.** The
  decisions that quote them are stamped with what they were measured on and
  with which half of each claim survives: the orderings do, the absolute levels
  do not, because every absolute distance in the study rose 16 to 45 percent on
  the new corpus. That stamp is the fix. Re-run one only if this stage needs
  its number, and if you do, say so in the report and update the stamped
  decision.

**STAMP THE PROVENANCE ON EVERY TABLE THIS STAGE WRITES.** Two columns or two
lines of metadata: the corpus label and the empirical weight rule. Stage 2h
could not answer "which corpus did this result run on" about its own output
without reconstructing it from file timestamps, and six of its results turned
out to be on a corpus replaced later in the same stage. This costs nothing and
it is the standing fix. `data/processed/CORPUS.json` names the active corpus
and `empirical.WEIGHT_RHO` names the rule.

**ONE THING THIS STAGE MUST NOT TOUCH.** `flip.FLIP_THRESHOLDS` is stale --
all three values fall outside their own recomputed intervals on the current
corpus -- and it is waiting on an author decision that belongs to Stage 3. This
stage does not need it. If something here reads it, say so in the handoff and
leave it alone.

**AND ONE THING THAT IS NOT THIS STAGE'S, recorded because a Stage 2h decision
wrongly assigned it here.** Decision 200 suggested Stage 2j re-open
`mode_share_alpha`, the generator's hump-share concentration. **That is a
GENERATION question and does not belong in a stage about method selection.**
Putting it here would be the "stage that does its own work plus a thin version
of another" failure the roadmap exists to prevent. Leave it; the author has been
asked where it goes.

## Stage 3 - figures

### What Stage 2j hands this stage, added 2026-09-29

**Stage 2j has run and it changes this stage's first task.** It measured the
mixed-method policy -- letting the UQ method vary BY MATERIAL rather than fixing
one method for all four materials of a pLCA -- against the true parents on the
same sixteen claims. `reports/STAGE_REPORT_2j.md` is the full account and is
short; read it before this section.

**FOUR THINGS IT SETTLED THAT THIS STAGE MUST CARRY.**

**One: there are TWO rules and only one of them is a method a reader can
follow.** The FEASIBLE rule keeps uniform weights throughout and switches only
the FAMILY -- a kernel estimate at or above the cutoff, a three-parameter
lognormal below. The KNOWN-SHARE rule switches the weighting too, and it needs
the true market shares, which nobody has. The feasible rule is worth **+2.9
percent pooled** over the sixteen claims against the best fixed uniform-weighted
method, beats the best fixed method claim by claim on **9 of 16**, median **+0.9
percent**. The known-share rule is worth 11.4 percent and is not a
recommendation. **Put the FEASIBLE rule on the scorecard as a seventh column and
not the known-share one**, because a column a reader cannot reproduce is worse
than no column.

**Two: the paper prints TWO RANGES AND NO SINGLE-DECLARATION CUTOFF, at either
level (decision 225).** The family split is **40 to 170** declarations and the
weighting split is **80 to 100**; `mixedpolicy.MIXED_THRESHOLD = 80` is a
constant the code needs to name one policy, not a result. Treat any `81` found
in a draft as stale rather than as a number to defend. The sweep
runs every 10 from 10 to 200 plus both degenerate ends, and everything from 40
to 170 is statistically
indistinguishable from the best; the whole sweep spans 0.73 points. **81 is a
FIT-level argmin and must never be printed as a claim-level threshold** -- that
is the conflation decision 163 already warns about, and Stage 2j is where the
claim-level answer was finally measured. Quote the range.

**Three: the weighting framing this file and four stages carried was WRONG, and
it is corrected by decision 212.** The study does NOT guess market shares from a
flat Dirichlet and compare that against ignoring them. At the shipped generator
setting each synthetic point carries `market[group] * within`, so the weight
mass on every product group equals that group's TRUE market share to 1.1e-16.
The contrast the paper draws is **what NOT KNOWING a market share costs**:
pooled error 23.24 percent with uniform weights against 20.25 with the shares
known, which is 2.98 points, or 12.8 percent of what was there. **Frame it as a
cost of missing information, never as a guess beating an omission**, and check
every caption and every sentence in this section against that.

**Four: the n_eff explanation is settled and the sloppy version is withdrawn
(decision 219).** Weights do not take observations away. Nine EPDs stay nine
EPDs; what the shares tell you is which population your nine are describing, so
a badly aimed sample is revealed rather than created. The bandwidth was tested
three ways -- the production `n_eff` rule, the plain count, and each fit's own
best bandwidth -- and the crossover survives all three, so it is not a bandwidth
artifact. The three-parameter lognormal, which has no bandwidth at all, shows
the same crossover. **A reviewer will find this counterintuitive, so the paper
owes the intuition in plain words, not just the three tests.**

**THIS STAGE'S FIRST TWO TASKS, in order.**

1. **Give notebooks 1 and 2 an `OUT` cell AND mark their twelve unmarked
   `savefig` cells.** There are TWO blockers and the first is the one this file
   used to get wrong: `audits/render_figures.py` raises on a missing setup cell
   BEFORE it checks markers, and neither notebook defines `OUT`, so marking alone
   unlocks nothing. `tests/test_render_figures.py` skips both notebooks today
   with the reason "has no setup cell". Clearing both gives seven-second figure
   rounds and is the prerequisite for everything below; notebook 3 is already
   done and is the worked example.
2. **Add the feasible rule as a seventh scorecard column** (decision 211) and
   fold Stage 2j's per-claim gains into it, then do the caption sweep described
   below.

**AND THE STAGE 2j FIGURE EXISTS AND IS CURRENT:**
`outputs/figures/CompareUQMethods_FIG_MixedPolicy.png`, drawn from
`TABLE_MixedPolicyThreshold.csv` and `TABLE_MixedPolicyRanking.csv`, needs a
slot in the figure numbering. It went through about a dozen rounds with the
author and the lessons generalise to Stage 3's own figures: **label every line
that is drawn** (three lines and two labels is unreadable); **separate a close
pair by pointing one label up and the next down**, not by nudging; **put
reference labels outside the panel** when the interior is crowded;
**colour-code a label to what it names and never reuse a curve's colour for a
band**; **mark anything hypothetical** -- the market-weighted lines carry an
asterisk and one footnote, because a reader who takes them for options misreads
the whole figure; and **a title should be a positive claim** that the panel can
actually support.

**TWO MORE THINGS 2j LEFT FOR LATER, both recorded and neither blocking.** The
group-composition split is computed for the known-share rule only, and
`MixedBackwards` -- market shares BELOW the cutoff, a control that should lose
to both recommended rules -- is in the code and has never been run. Both land
on the next full notebook run, whenever Stage 3 triggers one.

### Read this before anything else in this stage

**EVERY NUMBER IN THIS STAGE SECTION IS PROVISIONAL AND MOST OF THEM ARE STALE.**
This section was written across Stages 2b to 2h. Stage 2h then changed the real
arm's weighting rule and regenerated the synthetic corpus, and its own handoff
says every synthetic number in the paper moves. So a figure in this stage is not
allowed to take a number from this text. Recompute each one from the tables as
they now stand, and where a recomputed number differs from what is written here,
**the table wins and you say so in the handoff** rather than quietly using the new
one. Several numbers below are flagged individually where they are known to be
wrong; the flags are not a complete list.

**FIRST TASK: THE NOTEBOOK TABLES ARE CURRENT AND THE AUDIT SCRIPTS ARE NOT.**
The Stage 2h window established this when asked. The weight rule was ported at
2026-09-24 13:45, `corpus_2026-09-25` was generated at 11:27 the next day, and
notebooks 1 to 4 were re-run between 12:45 and 13:57 that afternoon. So every
table under `outputs/tables/` written by a notebook on 09-25 is current, and every
table under `outputs/tables/audits/` dated 09-24 is from the superseded corpus
and, if written before 13:45, from the old empirical weight rule as well.

**Six results are stale and each informs PROSE rather than a committed figure.**
None of them writes to `outputs/figures/` or to the top level of
`outputs/tables/`, so none can silently corrupt a figure. Re-run them, at a stated
cost of about two hours for all six: the judgment arm at about 25 minutes, the
upper-truncation sweep about 8, the generator sweep about 35, the bandwidth sweep
about 10, the certification credit about 20, and the bandwidth through the pLCA
about 30.

**Re-run them in this order, which is by how much the conclusion could move.** The
judgment arm first, because its headline is an ABSOLUTE error in a design
probability and absolute errors moved 25 to 35 percent with the regeneration. Then
the certification credit, for the same reason. The bandwidth sweep and the upper
truncation are comparisons BETWEEN methods on a common corpus and their orderings
should survive even as their levels move; re-run them anyway and report whether
the orderings held, because "should" is not a measurement. The generator sweep has
already been re-run and its result is in the Stage 2h answers.

**THREE THINGS ARE STALE IN A WAY THIS STAGE WOULD NOT OTHERWISE DETECT.**

1. **RESOLVED 2026-09-25, BEFORE THIS STAGE RUNS, and this item is no longer
   blocking. `flip.FLIP_THRESHOLDS` has been recalibrated by author decision.**
   It was a hard-coded constant calibrated on the superseded corpus whose three
   stored values all fell outside their own recomputed 95 percent intervals:
   0.0018 against a recomputed 0.00291, 0.011 against 0.01502 and 0.025 against
   0.03157, off by factors of 1.62, 1.37 and 1.26. The constants are now
   **0.0029, 0.015 and 0.032**, which are those recomputed values at two
   significant figures, and notebook 1 -- which reads them to turn a per-dataset
   weighting risk into a probability -- has been re-run, moving the mean
   probability that unknown market shares change which material leads from
   0.9870 to 0.9734 at the 1 percent level, 0.8642 to 0.8144 at 5 percent and
   0.7105 to 0.6524 at 10 percent. Notebook 3 still recomputes and prints the
   crossings beside the constant on every run, so future drift stays visible.
   **What is still open and belongs to this stage: no test compares the stored
   constant with the recomputed one, so the next drift will again be visible
   only to a reader of the notebook's output.**
2. **Every figure except the scorecard still carries the retired weighting
   labels**, because only the scorecard cell uses `display_method` and only the
   scorecard was redrawn. A figure showing "Variable" or "sampled market shares"
   is stale against the settled vocabulary and looks entirely fine.
3. **The scorecard figure's own CAPTION prose is stale even though the figure
   itself is current.** That is the next item, and it is this stage's clearest
   worked example of the failure it has to avoid.

**THE CAPTION IS THE WORKED EXAMPLE AND IT MUST NOT BE COPIED.** The Stage 2h
handoff's figure caption was corrected on the five rows whose definition changed
and left alone everywhere else, so it now mixes current and superseded numbers in
one paragraph while reading as a single consistent description. Against the
sixteen-row table the Stage 2h window supplied on the current corpus, the caption
is wrong on at least the following, caption value first:

    the design comparison, best method        12.0          ->  12.59
    the design comparison, range              12.0 to 17.1  ->  12.59 to 19.58
    a material's chance of being largest      32.0          ->  32.42
    a material's share of the total           10.0          ->  12.22
    the uncertainty index, range              43.6 to 45.5  ->  47.60 to 49.76
    the uncertainty index, best to worst      1.9           ->  2.16
    a cap's chance of saving 5 pct, range     33.0 to 58.1  ->  32.50 to 61.32
    the total's mean                          8.0           ->  9.43
    the total's standard deviation            22.1          ->  24.19

The caption also uses "sampled-share", which is retired vocabulary, in a caption
describing a figure that was redrawn after the vocabulary settled. **Rebuild the
caption from the table rather than repairing it line by line, and check every
other caption in the repository the same way.** A caption is the one place in a
paper where a stale number is invisible to every check this project runs.

**SECOND: the generator-parameter sweep was re-run and it is now a stronger
result than it was.** Stage 2h section 9's conclusion was measured before the
weight rule was ported and on the superseded corpus, so it was withdrawn and
re-derived. Fourteen configurations at two seeds against the new shipped
configuration and the reweighted empirical arm: nothing beats the default by more
than the weight-draw noise of 0.006 to 0.015, and the two configurations that used
to beat it, `mode_coupling` at 0.0 and 0.5, are now the two WORST of the fourteen
at +0.0277 and +0.0379.

**Write that inversion into the paper.** Section 9 had warned that those two
configurations won only because the cheapest way for the two halves of the study
to agree was for the synthetic arm to adopt the empirical arm's false assumption
that market share is unrelated to carbon intensity. Under one weight rule the
cheat stops paying and the same two settings become the worst available. That is
an independent confirmation of the stage's central finding, arrived at by a
measurement set up before the finding existed, and it is better evidence than the
port's own validation because nothing about it was chosen to come out that way.
The old shipped default, `min_q1_over_iqr` at 0.5, now measures +0.0188 and is
outside the noise, which is a second check in the same direction.

**And the author's hump-share proposal is no longer bad; it is neutral.** Moving
`mode_share_alpha` from 10 to 1 measures +0.0149 against a weight-draw noise of
0.006 to 0.015, so it sits at the edge and is not distinguishable from the
default, where the earlier measurement under the mismatched rules recorded it as
clearly worse. It costs nothing measurable and buys nothing measurable on the
objective. But it is separately the single most effective lever on the multimodal
SHARE, reaching 29.7 percent against a real 31.5, and the regeneration left the
conditional modality worse. **So it is a free choice on the objective and a
positive one on modality, and it belongs with the hump-spacing measurement below
rather than being treated as settled.**

**THIRD, AND IT GOVERNS EVERY NUMBER THE MANUSCRIPT QUOTES: say whether a
statistic is about ONE decision or about the AVERAGE of many.** Five of the
sixteen scorecard rows were computed the second way while their captions claimed
the first, and for those five the two differ by a factor of about twenty. The
study now carries both and each is right for a different reader: the error in one
decision is what a designer choosing between two options carries, and the error in
the average is the right quantity for a portfolio or a national stock model.
Every figure label, axis title and caption must name which one it is showing. Two
sentences in the manuscript reverse outright on this and both are marked below.

**FOURTH: the vocabulary changed at the close of Stage 2h and the figures are
where it will otherwise survive.** The two weighting schemes are **"market
weights"** and **"uniform weights"**, and the oracle scheme is **"known market
shares"**. "Variable", "sampled market shares" and "Dirichlet shares" are all
retired and must not appear in any axis label, legend, panel title or filename.
One sentence has to travel with the new label at first use in the methods: market
weights are DRAWN from a Dirichlet because production volumes are not published,
so the label does not mean real production volumes. Without that sentence a result
where uniform weighting wins reads as a modeling error, which has already happened
once in this project.

**Three framing points that must appear wherever the empirical arm is
described.**

**First, and most consequential: real ECC datasets are mostly unimodal to
look at.** Stage 2a-3 measured 95.3 percent of the empirical datasets as
having exactly one mode visible in a default-bandwidth KDE, while only about
half are unimodal by Silverman's critical-bandwidth test. The two tests are
answering different questions and the difference is not a detail: the usual
argument for a KDE over a parametric fit is that it can represent a second
mode, and on this evidence real ECC data rarely shows one. If a KDE still wins
in this study, the reason is skewness and tail shape rather than multimodality,
and the manuscript must say so rather than inheriting the multimodality
argument unexamined. Whatever Stage 2f finds about which modality measure
predicts W1 is what settles the wording. This is a substantive change to the
paper's motivation, not a caveat.

**The weighting result from Stage 2d, which is the strongest practitioner-facing
thing in the project.** Whether weighting matters is predicted almost exactly by
two numbers a practitioner already holds: the EPD count and the coefficient of
variation. The separation goes as about 0.73 * CV * n^-0.43, with an R2 of 0.99 on
both arms and nearly orthogonal contributions, 48 percent from size alone, 50 from
dispersion alone, 99.1 together. Set against the calibrated 5 percent flip
threshold, uniform weighting is safe only when the coefficient of variation is
below about 0.015 * n^0.43: 0.046 at ten EPDs, 0.120 at a hundred, 0.315 at a
thousand. The median real category sits at CV 0.63 and 47 EPDs and does not clear
it, and 91 percent of real categories fail the test. That is the paper's argument
for collecting market-share data, which is the one input that would remove the
problem rather than bound it.

Four things must travel with those numbers, in the same paragraph rather than a
footnote.

- **Every flip probability is an upper bound.** They assume four materials of
  equal material use intensity, which makes a ranking as fragile as it can be
  made. The continuous outputs do not carry that dependence, which is why they
  should lead. Stage 2e's group-size sweep will have moved the thresholds.
- **Every weighting number is a lower bound**, for the opposite reason. A flat
  Dirichlet understates the separation by 1.5 to 3.1 times when share clusters on
  products with adjacent coefficients, which is how real market share behaves.
  Matched-effective-sample-size tests show concentration alone is captured;
  coherence is not.
- **The location/shape split is not a recipe and must not be written as one.**
  The mean term carries a median of 0.725 of the uniform-to-variable distance,
  but the interquartile range is 0.457 to 0.933 and 28 of 147 categories are
  shape-dominated, with `CementGrout` at 0.024. Which way a category behaves is
  predictable from nothing measured, Spearman -0.11 against size and -0.10 against
  dispersion. And nobody can compute a market-weighted mean without already
  knowing the market shares, which is the premise of the companion paper. The
  finding says what kind of uncertainty unknown weights introduce, mostly
  uncertainty about a mean, which is the argument for production-weighted
  industry-average declarations.
- **This is the one place in the study where dispersion is first-order.**
  Everywhere else size is the only mechanism. Say so, rather than letting it read
  as a contradiction.

**The rule for printing any fitted crossing, settled in Stage 2h and superseding
the flat two-significant-figure instruction that used to sit here.** Every
crossing is fitted twice, once with a logistic and once with a
monotone-nonparametric fit that assumes only that the probability does not fall as
two models separate. The tables and the supplement carry BOTH values at full
precision with the bootstrap interval, because a reader checking the work or using
a constant downstream needs the unrounded number. **Prose and figure annotations
round at the first digit where the two fits disagree, and print both values where
they still disagree there.** It is not a fixed digit count: where the two fits
agree to four figures the prose may print four.

This matters because the published interval was a bootstrap interval on the
logistic's parameters, which says how well the data pin down that model and nothing
about whether a logistic is the right shape -- and the second fit lands outside
the published interval on five of the six constants. On the current constants the
rule gives:

    the safe lead at a 1 pct risk     2.1 against 2.2
    the safe lead at a 5 pct risk     1.64 against 1.61
    the safe lead at a 10 pct risk    1.5 against 1.4

No constant moves and nothing is thrown away; only the printed precision changes.
A reader who refits one of these with a different standard method gets a different
third digit, and printing that digit claims a precision the study does not have
while implying a cliff where the curve is smooth.

**The paper's contribution has changed shape, and the text has to change with
it.** Stage 2c established one mechanism, dataset size, with a threshold. The
manuscript argues for kernel density estimation as the flexible default; the
evidence supports something more specific and more useful. Three things follow.

- **The claim is a policy, not a method.** Use a KDE above about 80 EPDs and a
  three-parameter lognormal below.

  **Quote the threshold as 81 with a band of 68 to 106, and never as "about
  100".** The threshold itself is 81 on both the superseded corpus and the
  regenerated one, which is the strongest single piece of evidence the rule has.
  The band widened from 68-97 to 68-106 with the regeneration; quote the band from
  the current corpus. Stage 2f measured it properly: the datasets are resampled, the whole
  cost curve refitted, and each candidate threshold compared against whichever
  won on that same resample, so the interval is about the difference rather than
  the level. Eight independent bootstrap streams return 81, 68 and 97 every time.
  An earlier range of 59 to 134 came from a chosen tolerance rather than a
  measurement and is withdrawn.

  **Two ranges exist and they are different quantities. Do not quote one as the
  other.** The kernel and lognormal families change places at 46 to 70
  declarations, which is a property of the win-share curves and moves a little on
  a reseed, so it gets two significant figures. The best place to put a RULE is
  68 to 97, which does not move on a reseed. The rule sits above the crossing
  because the penalties are asymmetric: 4.8 points of extra error at a threshold
  of 24 against 6.5 at 304.

- **The two arms disagree about which fixed policy is better, and the text must
  reconcile it rather than quote whichever suits the sentence.** On the empirical
  arm out of sample, always-lognormal costs 4.9 percent over a per-dataset oracle
  and always-KDE costs 15.3. On the synthetic corpus, always-KDE costs 53.6
  percent and always-lognormal 178.5. That is not a contradiction, it is the size
  mix: the corpus holds 25 percent of its datasets above a thousand declarations
  where the real arm holds 7.4 percent, and the KDE's advantage is a
  large-dataset advantage. Post-stratifying to the empirical size mix is what
  reconciles them, and the reconciliation belongs in the text, because a reader
  who sees both numbers unexplained will conclude one of them is wrong. The
  threshold rule beats both on both arms, which is the point.
- **The flexibility argument survives in a narrower and more defensible form.**
  The KDE is the safer default, not the more accurate one: its worst single
  dataset is 1.61 times the best available method on real data against the
  lognormal's 2.75, and 6.9 against 24.5 on the corpus. Say that, and do not say
  it is the most accurate, because out of sample it is not.
- **Normal distributions are never the answer, which is the study's strongest
  negative result and is currently understated.** Across the four size bands the
  two normal fits take 0.0 to 8.8 percent of wins, running flat under 12 percent
  everywhere and effectively zero above a few hundred declarations. Most current
  practice fits a normal. Say it plainly.
- **Multimodality is not the mechanism and must stop being offered as one.** Stage
  2f confirmed it out of sample on 10,000 datasets, where the earlier evidence was
  in sample on 127: as the modality index rises the KDE's win share FALLS, from
  60.9 to 55.9 percent under equal weights, and under market-share weights it
  crosses downward from 67.0 to 46.5. Holding dataset size fixed does not rescue
  it. Silverman's critical bandwidth is flat. The advantage comes from matching
  general shape, skew and tail, not from resolving separate humps. The
  KDE's advantage is general shape matching, skewness and tail behavior.
  Structural materials look like the exception only because they are the
  well-populated ones, median n of 140 against 46 elsewhere; adding material tier
  to the size model contributes nothing detectable, p = 0.61. Report the material
  point as a consequence worth knowing, since the materials that dominate embodied
  carbon are the ones that clear the threshold, and not as a second finding.

**The corpus earned its place, and this is where to say so.** The real
categories cannot answer the question this paper asks, and the way that shows is
specific: refitting the same baseline model 20 times, changing only which
categories fall in which cross-validation fold, the synthetic arm reproduces its
R2 to three decimals, +0.280 and +0.524, while the real arm does not reproduce
its own sign, ranging -0.442 to +0.370 under equal weights and -2.605 to +0.337
under market-share weights. A negative value means the model predicts worse than
guessing the average. So any single number quoted from the real categories,
including a flattering one, is one draw from a three-point-wide distribution.
There are only about 150 real categories with enough declarations to analyze, and
that is too few. This is the justification for having built a synthetic corpus at
all, and it is an argument from measurement rather than from principle.

**Two corrections to that paragraph.** An earlier draft of it said "the 127 real
categories"; the arm is 147 datasets drawn from 138 queried categories and has
been since Stage 2a-3, so quote 147 or say "about 150" and never 127. And the R2
figures quoted just above were measured under the OLD weighting rule and on the
superseded corpus, so recompute them; the fold-to-fold instability is the claim,
not the specific values.

**And this argument is now stronger than it was, by an amount worth stating.**
The practitioner rule was calibrated on one synthetic corpus and came back
unchanged on a second, independently generated one with materially different
dispersion: the threshold is 81 declarations under both. That is a reproduction
across two corpora rather than a robustness check within one, and the paper
should make it as such. Note that the paper describes ONE set of synthetic data,
not two -- the superseded corpus is the less representative of the pair and
showing both would invite a reader to ask why the worse one is there. The
reproduction is evidence for the claim, not a second method to describe.

**The coverage limitation is now one category, not a general caveat.** The single
place the corpus fails to span the real range is `Aggregates`, a contaminated
database bin holding sinks, worktops and gravel together, at a coefficient of
variation of 6.93 where the rest of the real arm reaches 2.4. Removing every real
category above the corpus's range moves the KDE's win share from 40.2 to 40.5
percent and the size crossover from 124 to 122 declarations. So the limitation is
a statement about how the database files things, not a limit on the conclusions.
Recompute both figures against the current corpus.

**THE SENTENCE THAT USED TO CLOSE THIS PARAGRAPH IS WITHDRAWN.** It said widening
the corpus was investigated and declined because the best widening halved the
dispersion mismatch while multiplying the weighting-distance mismatch by 2.4. That
trade was real in every one of the 36 settings that found it and it was an
artifact of the two halves of the study weighting their data by different rules.
With both halves on one rule the trade disappears before the clustering knob is
turned at all, and the corpus was regenerated in Stage 2h: the dispersion distance
falls from 0.409 to 0.247 and the weighting-effect distance falls from 0.340 to
0.151 at the same time, with the overall match improving from 0.2216 to 0.1862.
**Do not write the declined-widening sentence into the paper in any form.**

**The unimodality figure must be restated with its bandwidth named.** "95 percent
of ECC datasets are visibly unimodal" is a property of Scott's rule, which
oversmooths this data by about 35 percent. At the bandwidth the study actually
fits it is 68.5 percent, and at that bandwidth the corpus and the arm agree within
a total variation of 0.078. Quote the bandwidth wherever the figure appears.

**Two smaller text rules.** "Lognormal" never appears without its parameter count,
because the two-parameter form is the worst of the four right-skewed families
while the three-parameter form is the best. And `TABLE_MethodCurves.csv.gz` at
91.7 MB needs handling before the deposit.

**One paragraph this stage owes the reader, and it is a credibility matter.**
Three of the four criterion changes Stage 2c made move numbers in the KDE's
favor: the bandwidth, the quadrature convergence and the tail term. Each was
chosen for an independent reason before its effect on the comparison was known,
and each defense is a convergence table anyone can recompute. Present them as one
paragraph about taking the criterion to convergence, not as three improvements.
Three separate improvements all helping the method under test reads badly however
sound each one is.

**The frame for the results section, established by Stage 2g.** A probabilistic
LCA answers five questions: what is the building's total, which materials
contribute most and how often, which materials contribute most to the
UNCERTAINTY, how effective is a reduction strategy, and is this design better
than that one. Every result in the study belongs to one of them. That frame is
what makes the study's main correction legible: "which material is the largest
contributor" is one of six numbers inside ONE of the five questions, not the
study's subject, which is how the paper currently reads it.

**Show every method on every claim, and do not compress the comparison into a
single spread.** The scorecard figure already does this and the text should
follow it rather than summarizing it away. Two reasons.

First, a claim can have a small spread because every method is right or because
every method is wrong, and only the best method's own distance from the truth
tells those apart. On a material's chance of being largest the best method is
about 32 percent out; on the uncertainty index nothing gets closer than about 44,
with under 2 points between best and worst. Those two look alike on spread alone
and are completely different situations.

**THE THIRD EXAMPLE THAT USED TO SIT HERE WAS WRONG AND IS THE REASON THE RULE
ABOVE EXISTS.** This paragraph previously read "on how often a specification cap
binds the best method is 0.5 percent out", and concluded that picking the right
method is nearly the whole problem on that claim. Stage 2h found that row was
averaging the error over many buildings before taking its size, so opposite-signed
errors cancelled. Per building the best method is about **30.6 percent** out and
the choice of method adds about **23**, so most of the error is there whatever you
choose. The sentence reverses. **A second sentence reverses with it**: "on what
using 25 percent less of a material saves, every method is exactly right" is true
of the average over many buildings and false for one, where every method is about
10 to 13 percent out. Both are in the manuscript and both must be rewritten, not
softened.

Second, and this is why a best-to-worst spread is the wrong summary: the paper
also concludes that a normal fit should not be used. Once that is said, a spread
whose worst member is a normal is measuring the cost of a choice nobody should
make. **Where the paper needs a cost-of-choice number, compute it among the
methods actually in contention** -- the kernel estimate and the three-parameter
lognormal under each weighting scheme -- and report the normal's distance
separately as the size of the error being avoided. The two are different claims
and collapsing them overstates how much the remaining choice matters.

The stronger move is not to summarize at all. The full matrix shows that the
methods separate on some claims and not others, that the ordering inverts with
dataset size, and that on two of seven per-material metrics no method separates
from the runner-up. A binned headline hides all three, and all three are the
paper's actual contribution.

**Demote the rank-1 frequency, with four independent arguments and the strongest
stated first.** It is the worst-recovered of seven per-material numbers for EVERY
ONE of the six methods, without exception, and under a normal fit its error
exceeds the entire real spread of that number between materials, at 1.037 and
1.074, meaning the ranking carries no information about which material is which.
The other three arguments are that it is fragile when materials contribute
equally, that it carries a 3.67 percent noise floor from an arbitrary tie-break,
and that it is only 9 percent predictable from a material's own data because it
is a property of the group rather than of the dataset. **Report it as one
statement among five, quoted with its noise floor**, because it is what the
manuscript currently leads with and readers will look for it.

**Lead instead with the pair: how much, and how uncertain.** A material's
estimated contribution recovers at 0.51 to 0.71 of its own between-material
spread, and the spread of that contribution at 0.42 to 0.50, the best of the
seven candidates. They are not substitutes and no one method is best at both.
A reader given only one cannot tell a material that is big from one that is
uncertain, and those call for different actions: the first is a design problem,
the second a data-collection problem.

**The instability is not an artifact of this study's construction, so report it
plainly rather than defensively.** Every material here is normalized to the same
average and given the same use intensity, which makes a ranking as fragile as it
can be, and that is a fair objection. But Marsh et al. (in press) find the same
thing on a real four-option staircase with real quantities: the top-contributing
product changes with the uncertainty characterization scenario. It is a property
of ranking near-equal contributors, not of synthetic data.

**The normal's decision-level failure leads the case against it.** Everything
else in the project measures how far a normal fit sits from the truth. The
design comparison measures whether it picks the right design, and at a claimed 0
percent saving one of the two normal fits is wrong 65.8 percent of the time,
which is worse than guessing. A systematically optimistic method lands on the
wrong side of a near-tie more often than chance. Lead the normal discussion with
that, because it is a failure a reader can feel, and follow it with the scoping
below.

**Scope the normal advice rather than stating it flat.** Neither normal fit is
ever best on any of the sixteen claims, and the normal is never the best method in
any size band under either weighting, but the penalty spans a wide range.
**The specific penalties below are stale on two counts** -- they predate both the
correction to the five reduction-strategy rows and the regeneration -- so
recompute them; what is known to have moved is that the most expensive single
claim is now a cap's chance of saving 5 percent at about 25 points of
cost-of-choice, where it used to be how often a cap binds at about 31. The shape
of the old numbers, kept here only so the recomputation has something to check
against: 27.0 and 24.9 percent behind the best on how often a cap binds and what a
cap's chance of saving 5 percent is, 14.1 on a material's chance of being
largest, and within one point of the best, and not the worst of the six, on a
material's 95th percentile, its share at the building's 95th, and the uncertainty
index. What survives everywhere is the bias: on the three numbers where a signed
error means something the normal is the most biased of the six, at +4.8, -10.5
and -28.5 percent of true level against the kernel estimate's +0.3, -4.5 and
-14.1. Note also that on a share or a rank frequency the signed error is
identically zero for every method because the four values sum to one, which is
arithmetic and not evidence of unbiasedness.

**Promote the uncertainty index; do not introduce it.** An earlier claim in this
project's own record, which reached this prompt too, said the study reports it
nowhere. That is false: the manuscript reports it in two panels of Figure 5,
defines it in the supplement, and draws a conclusion from it. What is true is
narrower, that it appears as a secondary observation about method agreement
rather than as one of the five questions. And it must be reported with both
numbers together: it has the lowest disagreement between methods of any output,
0.5035 against 1.042 for the rank frequency, AND every method is out by about
half its own between-material spread, 0.508 to 0.531. Nine tenths of that error
is common to all six methods and comes from estimating a variance from few
declarations, so no choice of method repairs it. A number the methods agree about
and are all wrong about is the one thing a paper must not present as reliable.

**State the tension between accuracy and tail-robustness, which is now half
settled.** The metrics that recover the truth best are levels, and levels are
exactly what a thin far tail wrecks; the ones immune to it are shares, and they
recover worse. What makes the levels safe to report is that the study's criterion
charges for the thing that wrecks them, through the tail term.

**And Stage 2h measured a remedy that the paper should offer the reader even
though the study does not adopt it.** Capping every fitted model at two to three
times the largest value actually observed cuts the worst runaway fit from 5.4
times the data's own spread to 1.8, and against the known truth it is not a cost:
the difference lands in the fourth decimal, and the head-to-head gap between the
two leading methods barely moves. It does not pull the two halves of the study
apart either, at 0.12 of a percentage point. It was NOT adopted, because adopting
it would move every number in the study a second time for a failure two existing
safeguards already keep out of the results, and the author decided that. **So the
paper states it as a one-line remedy a reader facing extreme values can apply, with
the measured figures behind it**, and the code stays in the repository tested and
unused. Capping at the largest observed value itself is clearly harmful and the
paper should say two to three times, not one.

**The limitation paragraph must be rewritten around a different characteristic,
and the word "structural" must come out.** The paper's stated limitation is that
the corpus cannot reach the dispersion of the most variable real categories. That
is no longer the dimension on which the two halves sit furthest apart. After the
Stage 2h regeneration the worst-matching characteristic is **how lognormal the two
arms look, at a distance of 0.341**, having been 0.233 before; dispersion fell from
0.409 to 0.247. Write the limitation around the new worst characteristic, and say
why the trade was worth taking: matching on "how lognormal it looks" was measured
to move the comparison between the two leading methods by 0.001, against 0.026 to
0.043 for dispersion, so the study spent a characteristic worth a thousandth to buy
one worth several hundredths.

**The joint modality-and-dispersion gap stays as a limitation, its framing
changes, and it may have got WORSE rather than better.** What must not be written
is that the gap is structural. That claim rested on a trade which held across 36
settings and three levers and which turned out to be an artifact of the two halves
weighting their data by different rules.

**And the regeneration half-addressed this gap: the marginal improved and the
conditional got worse.** Measured on the current corpus on the weight-invariant
unweighted dispersion, with "dispersed" meaning above the real arm's own upper
quartile:

                         multimodal   dispersed   both   multimodal GIVEN dispersed
    real                     0.246       0.254   0.054              0.212
    corpus, superseded       0.241       0.028   0.006              0.221
    corpus, CURRENT          0.216       0.081   0.011              0.141

The both-at-once shortfall roughly halves, from the real arm holding 9.0 times the
corpus's share to 4.7 times, and dispersion alone improves from 9.1 times short to
3.1. **But the conditional reversed.** Among dispersed datasets the corpus used to
be multimodal 22.1 percent of the time against the real arm's 21.2, which matched,
and it is now 14.1 percent against 21.2. The multimodal share slipped from 0.241
to 0.216 against a real 0.246. That is the same trade the hump measure shows
marginally, seen jointly.

The reason is mechanical and known: the regeneration closed the dispersion gap by
widening the truncation bound and raising the dispersion target, which slides the
whole distribution toward zero rather than separating humps, and an earlier stage
established that this is exactly why separating humps costs spread. The two levers
built for the other mechanism, `genconfig.separation_dispersion_frac` and
`genconfig.shoulder_frac`, are confirmed still at 0.0 in both the live
configuration and the configuration recorded inside `corpus_2026-09-25`, so
neither has ever been measured under the settled weight rule.

**So measure the hump-spacing fix in this stage, under the settled weight rule and
on the current corpus, and report it against the weight-draw noise of 0.006 to
0.015.** The criterion is the one that has always governed it: adopt the change
only if it closes the joint-structure gap without making the corpus overstate the
paper's headline weighting effect relative to the real arm. That trade was the
reason for rejecting it and the trade is now known to have been an artifact, so
the measurement is genuinely open rather than a re-run of a settled question. If
it still fails, report both measurements and leave generation alone; either result
is publishable and the limitation paragraph is written from whichever it is.

What the gap does not show is that the answer would change: on those datasets, in
both halves of the study, the kernel estimate does no better, so it is a coverage
gap and not a hidden result. Recheck that on the current corpus before repeating
it.

**The single most important explanatory result in the project, and the text has
to carry it.** Stage 2e measured how the methods fail, and the mechanism is bias
rather than imprecision. A material's estimated contribution is biased high by the
normal, +0.044 uniform and +0.050 variable, low by the lognormal, -0.038 and
-0.027, and almost not at all by the kernel estimate, -0.014 and +0.003. On a
single material a bias of 0.04 vanishes into noise six times larger. But bias adds
across the materials of a building while the random part cancels, so the bias is
what survives: the normal overstates a four-material building by 4.4 and 5.0
percent, the lognormal understates it by 3.8 and 2.7, and the kernel estimate
stays within 1.4 either way. The same percentages would hold for a twenty-material
building while the random error shrank toward nothing.

So the choice of method shifts a whole building's estimate by up to nine
percentage points, systematically, and adding materials will not average it away.
A method that leans is a worse problem at building scale than one that is merely
imprecise. That sentence is the reason the paper's recommendation is about
accuracy of shape rather than about goodness of fit, and it should appear early.

**Two figure conventions from Stage 2e that must not be quietly undone.** In the
dominance figure the upper panel is a rolling mean and the lower a rolling median,
because a flip rate is a mean of zeros and ones while the change in a contribution
is heavy tailed; reading a mean against a median there made a flat panel appear to
rise. And both panels are held at four materials rather than pooling group sizes,
because a Dirichlet draw over twelve materials produces large top-two ratios far
more often than one over two, so pooling fills the right of the axis with
twelve-material groups and produces a group-size effect wearing a dominance label.
Neither is presentation; both are correctness.

### Five results blocks Stage 2h produced that the manuscript does not yet have

These are new findings, not caveats. Each needs a figure slot or a table, and the
first two are results the paper's audience will care about more than anything
else in it.

**1. The judgment-driven methods, placed on the same axis, and the finding is not
the one anyone expects.** The pedigree matrix is what the main life-cycle
databases apply and what a practitioner reaches for with no dataset. It cannot be
compared like for like with a data-driven fit, and the paper must not claim it
can; what makes the comparison possible is that the yardstick, error against the
true distribution the data came from, does not care how a model was built.

Two dimensions were swept and the second is the one that matters. A pedigree model
is a SPREAD applied around a POINT ESTIMATE the practitioner already holds, and
nothing puts that point where the category's true average sits. At the fit level a
judgment-driven model is 2 to 100 times further from the truth than the best
data-driven fit, and the realistic case -- the point taken from one declaration the
practitioner happened to obtain -- is 4 to 12 times worse. **At the decision level
a well-centered one is competitive**: on "is option B better than option A" the
six data-driven methods are out by 0.080 to 0.114 and a correctly centered
pedigree model by 0.097, holding at 0.097 to 0.121 across a six-fold range of
spread. **What breaks it is the center, and how it is wrong matters more than how
much.** A displacement applied to every material alike cancels exactly, because
both designs' totals scale together; displaced independently per material it costs
0.111 at 10 percent, 0.159 at 25 and 0.223 at 50, and the realistic one-declaration
case costs 0.170 to 0.271. Shape matters too: at matched spread and correct center
the pedigree lognormal is out by 0.097 where a uniform is 0.206 and a triangular
0.198, which is the measured reason uniform and triangular belong in this arm
rather than the main comparison.

**And the sourcing gap is closed, with a result that is stronger than the paper
currently dares.** The author supplied the pedigree matrix's factor table and all
3,125 score combinations were enumerated from it: best possible scores give a
spread factor of 1.025, a middling combination 1.242, worst possible 1.587, against
a typical real material category at 1.871. **A pedigree model is systematically
NARROWER than the data it stands for, and 61.9 percent of real categories are wider
than its worst possible score.** State plainly that this is not a defect in the
matrix -- it answers how uncertain ONE number for ONE process is, not how much
products within a category differ from each other -- and that the two are therefore
not rival estimates of the same thing. Then state the consequence, carefully,
because it is the sharpest practitioner-facing sentence in the paper: the judgment
route understates uncertainty on precisely the quantity it exists to express.

One arithmetic trap is recorded because this project walked into it: every factor
in the published table contributes to the SQUARE of the spread, so a model quoted
as a spread halves the exponent, and quoting the combined factor directly doubles
it. The published range is computed for a building material; a different flow type
shifts it slightly.

**2. The certification credit, which is the paper's clearest answer to "so what".**
Green-building certification awards points for a demonstrated reduction against a
baseline, typically 10 percent, and under a probabilistic LCA that becomes
"demonstrate a 10 percent reduction with 75 percent confidence". The study had
scored the error in the probability but never the DECISION the probability is used
for, which inherits that error AND a cliff at the threshold. Over 3,000 cases the
truth earns the credit 17.4 percent of the time, at least two of the six methods
disagree on 18.2 percent, the best method calls it wrong on 8.8 and the worst on
12.5. **The fragility is the threshold, not the methods**: the six disagree on 65.3
percent of designs within 0.05 of the 75 percent line, 46.3 percent between 0.05
and 0.10, 20.5 out to 0.25 and 3.4 beyond it. A design comfortably over or under
the bar is called the same way by every method and a design sitting on the bar is a
coin toss. Write it as an argument for stating the margin, not against writing
credits this way. A normal is the worst method on 8 of the 15 tier and confidence
combinations tested.

**THE SCHEME IS SOURCED AND THE TIERS MATCH, BUT ONE THING MUST NOT BE
ATTRIBUTED TO IT.** LEED v4.1 BD+C, credit MRc1 Building Life-Cycle Impact
Reduction, Option 4, awards points on a whole-building LCA of the project's
structure and enclosure against a baseline building: Path 1 is performing the
assessment at all, for one point; Path 2 is a minimum 5 percent reduction in at
least three of six impact categories, one of which must be global warming
potential, for two points; Path 3 is the same at 10 percent, for three points; and
Path 4 is 20 percent on global warming potential plus 10 percent in two other
categories together with reuse or salvage, for four points. On Paths 2 through 4
no impact category may increase by more than 5 percent. LEED v4 carried a flat 10
percent in three of six categories. So the study's 5, 10 and 20 percent tiers are
the scheme's own tiers and the paper may say so.

**Cite the USGBC credit library entry for MRc1 in LEED v4.1 BD+C, not a
consultant's summary**, and put it in `refs/`. Every source found in a quick
search was a software vendor or a consultancy restating the credit, which is not
what a Building and Environment reader should be sent to.

**THE FRAMING, WHICH THE AUTHOR HAS SET: this is a hypothetical, and writing it
as one is what makes it a contribution.** LEED states a deterministic threshold
and attaches no confidence level, because it does not contemplate a probabilistic
assessment. The paper's premise is that IF probabilistic LCA were incorporated
into a scheme like LEED, the scheme would have to attach some confidence
requirement to its existing tiers, and 75 percent is a plausible choice. So the
tiers are LEED's and the confidence level is the paper's, and the sentence
introducing this has to make that split explicit -- not as a disclaimer but
because the finding IS about scheme design. What the study then shows is that the
choice of confidence level is not a free parameter: at any level, designs sitting
near the line are called differently by different UQ methods two thirds of the
time. That is an argument for whoever writes such a rule to state a margin rather
than a bare threshold, and it is a more useful contribution than reporting that an
existing deterministic rule is fragile, which it is not, because a deterministic
rule has no methods to disagree.

Two further caveats belong with it. LEED's reduction is measured on structure and
enclosure against a baseline building across at least three impact categories,
where the study measures whole-building global warming potential between two
designs, so the mapping is an analogy and the paper should say which parts carry
over. And the scheme's own 5 percent no-increase constraint on the other
categories is a second threshold with the same cliff, which the study does not
measure and should not imply it has.

**A defect in the study's own code was found here and it changes no published
number.** The existing comparison margin computes the share of simulations in
which the proposal comes in below the baseline times a margin, and the study's
margins of 1.0, 1.05 and 1.2 are all above one, which asks "is the proposal better,
or worse by less than 20 percent". That is a tolerance, and a credit needs the
margin below one. At a true 20 percent saving the 1.2 margin reads 0.9993 against
a plain "is it better" of 0.9628: looser, not stricter. The code comment described
it the other way round. The 1.05 and 1.2 figures are what the comparative-LCA
literature reports and are correct AS TOLERANCES; **the sentence describing them is
what was wrong and that sentence is in the manuscript.**

**3. A per-category weighted number carries nine times the uncertainty of the
arm-wide one, and one published R2 is a casualty.** Across 25 independent redraws
of the real arm's market shares, the central weighting quantity has a typical
spread of 0.046 on a median of 0.108, which is 47 percent, and a worst range of
1.37; arm-wide, the median of the same quantity is 0.0981 plus or minus 0.0052,
which is 5 percent, and the two exponents of the practitioner rule are stable to 5
percent. Every statistic that does not use the weights moves by exactly zero, which
is the control. **So a statement about one material category has to be given as a
range and a statement about the collection is safe as a number**, and the paper
does not currently distinguish them.

**The casualty.** The rule predicting how much weighting matters from two numbers
a practitioner already holds is published with an R2 of 0.991, computed on the
median over a thousand draws. On a single draw it explains 0.824, with a
correlation against dispersion of 0.573 against a published 0.731. **The
characteristic stored in the published data table IS a single draw**, so a reader
recomputing the rule from the deposited table will not reproduce 0.991. Either
publish the median-over-draws characteristic alongside, or state the single-draw
figure as the reproducible one and the 0.991 as the population value. Do not
publish 0.991 against a table that yields 0.824.

**4. Two more families, and the family list is no longer short.** Gamma and
Weibull were added. Weibull ranks below gamma, and gamma is already established as
indistinguishable from the three-parameter lognormal, so the ordering of the
paper's own methods is untouched. This retires the "only two families were tested"
objection outright and belongs in the supplement unless it crowds the main results.

**5. One of the sixteen scorecard rows is redundant and the paper must report one
of the two, not both.** "What using 25 percent less of a material saves" and "a
material's share of the total" now carry identical numbers in all six cells, and it
is an identity rather than a coincidence: removing a quarter of a material's
contribution makes the error in the first exactly 0.25 times the error in the
second, so their relative errors are equal to machine precision. Verified across
60,000 rows at a correlation of 1.00000000. Under the old averaged definition both
rows read 0.00 and the identity was invisible. Drop one row from the figure and say
in the caption why.

**`FIGURE_STYLE.md` is binding on every figure in the repository, not only on
new ones.** Two figures comply. The rest of notebooks 1, 2 and 3 predate the
guide and none has been checked against it. `src/figstyle.py` supplies the
palette, the rcParams, direct labelling and an automatic text-overlap check.
Bring all of them to it and report any figure the guide cannot accommodate rather
than exempting it silently.

**Fix the overlap detector before trusting it.** Stage 2g found it had never
checked a panel TITLE: the plotting library keeps a separate text object for a
left-aligned title, the detector read the centered one, and this project sets
every title left-aligned. It reported no overlap on a five-column figure whose
titles plainly overlapped. A clean report from that tool is not evidence until
this is fixed and tested.

**Four other housekeeping items.** One results table is 96 MB and needs handling
before the deposit. Stage 2e fixed the Unicode minus for figures that call the
style module; older figures still carry it, and the author's output is plain
ASCII. British spellings survive in the style guide itself, which uses "colour"
throughout, in earlier decision-log entries, and in six cells of the third
notebook; this stage owns that sweep. And one committed figure is now a redraw
behind its own code: the supplement figure of example datasets by size band drew
each curve across the full range its bounds allow while the panel shows only the
body, so for a typical panel as few as 34 of 600 curve points landed inside the
frame and the shading collapsed. The code is fixed and the image is not, because
the first notebook cannot be driven by the fast figure tool. Re-running that
notebook redraws it and moves nothing else; doing so also closes the corpus
characteristic-list omission, which needs the second notebook rerun and a fixture
re-frozen.

**Do not act on the repository history.** It stands at 2.5 GB against the 391 MB
recorded when its size was accepted, the growth being large tables re-stored
whole on every commit. The rewrite that would shrink it was declined because it
would break the published deposit. The number is recorded so it is somewhere, not
so a later session acts on it.

**Two figure decisions carried from Stage 2b.** In the
W1-against-characteristic figures the normal's W1 blows up on high dispersion and
sets the y-axis, compressing the KDE-versus-lognormal difference into the bottom
fifth of each panel; use a log y-axis. And in
`CompareUQMethods_FIG_RankVsDatasetSize.png`, the middle row showing the SIZE of
the gap rather than the rank is the one the paper should use, because a mean-rank
curve turns an 8 percent gap at n = 3-9 and a sevenfold gap at n >= 1000 into the
same picture.

**The coverage claim has to be restated, and its figure rebuilt.** The
manuscript says the corpus covers the empirical characteristic space and extends
beyond it on every side. That was true when measured, on the Stage 2a arm whose
maximum coefficient of variation was 2.40, and it stopped being true when Stage
2a-2 rebuilt the arm from raw values. Stage 2b then removed two contaminated
categories and two mislabelled records, and the shortfall is now **5 uncovered
dataset-metric pairs out of 1,470**. **Decided: option A in discrepancy entry 34** - state coverage as
measured and name the exceptions. Option C was rejected as circular. **Option B's premise is
partly withdrawn and the text must not cite decision 48's numbers.** It was
ruled out on the grounds that no generator parameter reaches an empirical
coefficient of variation of 14.34, but that figure was inflated by two
mislabelled `PowerCabling` records, one declaring 14,300 kgCO2e for a metre of
building wire. The real maximum is 6.93, `Aggregates`, against a synthetic 2.58,
so the gap is about half what decision 48 assumed. It is still a gap and option A
still stands, but state it from the rebuilt tables.

So: rebuild `outputs/figures/CompareUQMethods_FIG_MetricCoverage.png` against the
147-dataset arm and the active corpus, mark the uncovered points rather than
hiding them, and write the claim as coverage with margin except at the extreme
upper tail of dispersion and above 9,999 values per dataset, where the probe set
covers by design. Three of the `n` failures are the large `ReadyMix` strength
classes and are decision 19 working as intended.

**Second, some EC3 categories were not single product populations and have
been split.** With the right tail no longer cut, `PowerCabling` spanned about
1.7e-05 to 242 times its own mean, `Aggregates` 3.2e-06 to 265 and `Insulation`
3.0e-04 to 99, with coefficients of variation of 13.4, 10.3 and 7.8 against a
median of 0.78. Stage 2a-3 resolved them with three metadata rules: drop the
categories that are non-leaf nodes of EC3's category tree, because those hold
the EPDs EC3 did not place in any child and are residual bins by construction
rather than products; split concrete by specified 28-day compressive strength;
split insulation by material type. Stage 2b then dropped `Chairs` and
`Grouting`, both EC3 leaf categories that no tree rule could catch, because 15 of
86 and 44 of 225 of their records name the product the category claims. The arm
is **147 datasets drawn from 138 queried categories**, and both numbers belong in
the text.

**Describe this as a question of what a dataset means, not as a dispersion
fix.** A dataset here stands for one material choice in a probabilistic LCA, so
the test is whether a category is something a specifier could name. Splitting
`ReadyMix` by strength class moves its coefficient of variation only from 0.29
to 0.27 and is still right, because 4,000 psi and 5,000 psi concrete are
different products. Getting this framing wrong is what cost Stage 2a-3 two
false starts. State plainly that no category was split on its ECC values,
because splitting on the distribution while measuring the distribution would be
circular; the coefficient of variation was used only to decide which categories
to examine. Note also that EC3 records carry no usable subcategory field, which
is why product type could not be used directly, and that four residual parents
were kept because none of their children are in the arm. The unsplit arm is a
Stage 2h sensitivity, which is the evidence the resolution did not manufacture
the result.

**Third, and separately:** The synthetic corpus was calibrated against the 147 empirical
datasets in Stage 2a: the coefficient-of-variation centre, the component
overlap bounds and `position_skew` were all chosen by scoring candidate
configurations against the empirical characteristic distributions. That is the
right way to build the corpus, and it is also why the empirical arm is **not an
independent test set**. The honest statement is that the synthetic data were
constructed to resemble real ECC distributions on their statistical
characteristics, so agreement between the arms demonstrates consistency rather
than out-of-sample validation. Say it plainly in the text and in the Table 1
caption. Unstated, a reviewer will call it circularity; stated, it is ordinary
calibration.

*One window, at the end.*

Read `CLAUDE.md` at the repository root for project context before
starting. Then commit any uncommitted work on the current branch
with a clear message, then create and check out a new branch named
`stage-3-figures`. Do all of this stage's work there. Commit as you go, in
logical units, so I can bisect later if a number moves unexpectedly. Tell
me the branch name and the commit you branched from.

Regenerate figures with these changes. Keep a style module so fonts,
colors and sizing are set in one place.

**First, absorb the Stage 1 work that was deferred to here.** Stage 1 stopped
before Phase 5 and most of Phase 6, so this stage inherits them, and Phase 5
comes first because everything else depends on it:

- **Phase 5: compute and plotting are separated.** Figures read only from
  tables on disk, never from in-memory state. Stage 1 left several figures
  reading kernel state, which violates a standing constraint and is why the
  pLCA figures were unreproducible in the first place. No figure in this stage
  may take its data from anywhere but a persisted table.
- **Phase 6 remainder:** reconcile the README's output list with what the
  notebooks actually write, and reduce the remaining oversized figures. The 15
  orphaned figures Stage 1 left in place are handled by the manifest process
  below rather than deleted outright.

- Merge current Figures 2 and 3 into a single figure. The strip plot of W1
  distances with mean labels occupies the left half; the rank frequency
  heatmap occupies the right half. Generate it from one function writing
  one file, not by stitching two existing PNGs.
- Rebuild Figure 4. Cut to the metrics identified in Stage 2f. Lay it out
  as a grid where the uniform and variable versions of the same metric sit
  side by side in a row, so they compare at a glance. Order rows by
  importance, least interesting first, so the figure can be walked through
  in the same order as the text.
- Also produce an alternative version of that figure plotting the
  difference in W1 between the uniform and variable version of each
  estimation method, giving three curves instead of six. I want to compare
  both versions before choosing.
- In the pLCA scatter figure, drop the highlighted single-pLCA points.
  Label panels explicitly as (a), (b), (c), (d).
- Every figure needs axis units, and any figure whose colors or line styles
  carry meaning needs a legend that stands alone without the caption.
- Report every aggregate that appears in a figure with its confidence
  interval. Stage 2e attaches bootstrap intervals to the headline percentages
  and NRMSE values; Stage 2d produces intervals on the flip-probability
  threshold crossings; Stage 2c produces regret distributions. Use whichever
  applies to the quantity plotted.

### Figure file hygiene

Figure filenames in the repository have drifted. Some main-text figures are
named SUPP, some supplements are named FIG, several have no number, and
there are likely stale PNGs that nothing generates any more.

Build a manifest before changing any filename. For every image file in the
repository, record the file, the code that generates it (notebook and cell,
or module and function), and where it appears in the manuscript. Any file
with no generator is a stale candidate; any generator writing a file the
manuscript never uses is also a candidate. Show me the manifest and the
candidate list, and move candidates to an `archive/` directory rather than
deleting. Do not delete anything without asking. Some of these files belong
to a published Zenodo deposit, so renaming happens in the working
repository only; the deposit is refreshed at submission.

Then adopt this convention, and have the generating code write the names
directly so filenames cannot drift again:

    CompareUQMethods_FIG<N>_<ShortDescription>.png
    CompareUQMethods_SUPP<N>_<ShortDescription>.png

Getting the number exactly right matters less than getting the FIG versus
SUPP token right and having every file carry a number. Current state, with
the mislabeling flagged:

| Manuscript slot | Current filename | Problem |
|---|---|---|
| Figure 1 | CompareUQMethods_FIG_PDFandCDFofUQMethods.png | no number |
| Figure 2 | CompareUQMethods_FIG2_Wass1DistStripAndRank.png | correct, but being merged with Figure 3 |
| Figure 3 | CompareUQMethods_FIG_WassRankFreq_Heatmap.png | no number, and being merged into Figure 2 |
| Figure 4 | CompareUQMethods_SUPP_WassDistanceVsMetric_ALLMETRICS.png | labeled SUPP but is a main-text figure, and no number |
| Figure 5 | CompareUQMethods_FIG_ScatterPlot_UQResults_Subset.png | no number |
| Figure 6 | CompareUQMethods_FIG6_PLCAVisualizeUQFits.png | correct |
| Figure 7 | CompareUQMethods_FIG7_RanksByDatasetAndPEWT.png | correct |
| Supplement 1 | CompareUQMethods_FIG_DemonstrateDataGeneration.png | labeled FIG but is a supplement, and no number |
| Supplement 2 | CompareUQMethods_SUPP_GeneratedVsEmpiricalMetrics.png | no number |
| Supplement 4i | CompareUQMethods_SUPP_ScatterPlot_UQResults_All.png | no number |
| Supplement 4ii | CompareUQMethods_SUPP_AllResultsByAllUQMethods.png | no number |
| Supplement 5i | CompareUQMethods_FIG2_KSTestStripAndRank.png | labeled FIG2 but is a supplement |
| Supplement 5ii | CompareUQMethods_FIG2_Wass2DistStripAndRank.png | labeled FIG2 but is a supplement |

Merging Figures 2 and 3 shifts everything downstream up one number. Propose
the new numbering and let me confirm before renaming. New figures from
Stage 2 also need slots: the coverage figure from 2a, the flip-probability
curve from 2d, the regret distribution from 2c, the claim scorecard from 2g as
corrected in 2h, the judgment-arm comparison and the certification-credit
disagreement curve from 2h, and the per-material policy comparison from 2j,
which has run and whose figure is
`CompareUQMethods_FIG_MixedPolicy.png`. **There is no anchor building and no Stage 2i figure**: 2i was decided
against and the real-building anchor comes from citing Marsh et al. (in press),
whose Concrete-Precast staircase supplies the one real top-two contribution ratio
the study uses. A later note in this file listing 2i as optional is stale; treat
it as closed.

Two more requirements. Write every figure at publication resolution and
emit a vector version (PDF or SVG) alongside each PNG. And add a check that
fails loudly if two generators try to write the same filename, since that
is probably how the duplicate FIG2 names above happened.

Record the final convention and the figure manifest in CLAUDE.md.

---

## Stage 4 - repository and deposit

*Required, at the very end.* Not optional: the code is cited in the paper as
a public Zenodo deposit, so the deposit has to match what the paper describes
at submission.

Read `CLAUDE.md` at the repository root for project context before
starting. Then commit any uncommitted work on the current branch
with a clear message, then create and check out a new branch named
`stage-4-deposit`. Do all of this stage's work there. Commit as you go, in
logical units, so I can bisect later if a number moves unexpectedly. Tell
me the branch name and the commit you branched from.

**Note for the manuscript rewrite, which happens outside Claude Code.**
Several Discussion paragraphs in the current draft exist to excuse
limitations that this rework removes: only two parametric families tested,
one KDE bandwidth rule, one lognormal fitting method, no out-of-sample
evaluation, and a pLCA construction designed to amplify the effect. Once
the sweeps have run, those paragraphs invert from caveats into findings.
When each stage finishes, note in CLAUDE.md which manuscript limitation it
retires and what the replacement claim is, so the rewrite has a checklist
rather than requiring me to reconstruct it from memory.

**Three items Stage 2h added to this stage.** The British-spelling sweep now
also covers `FIGURE_STYLE.md`, which uses "colour" throughout, earlier
decision-log entries and six cells of the third notebook. The superseded synthetic
corpus sits on disk beside the current one, as every corpus in this project does,
and the deposit must make clear which one the paper describes and that the other is
kept for diffability rather than as a second dataset. And the retired vocabulary --
"variable", "sampled market shares", "Dirichlet shares" -- has to be swept out of
column names, filenames and the README, not only out of the figures.

**RENUMBERING THE FIGURES IS NOT THIS STAGE'S, added 2026-10-01 by the author.**
"Don't worry about figure renumbering yet. That'll depend on what we end up
including in the manuscript. That will be one of the last things we do." Stage 3
proposed a numbering and did not apply it; it stays unapplied until the
manuscript's figure selection is settled. Decision 235. The figure manifest at
`outputs/tables/audits/TABLE_FigureManifest.csv` is still what the README
needs, and it does not depend on the numbering.

Update the README and repository structure for re-deposit to Zenodo at
submission. The code is cited in the paper as a public artifact, so it
needs to be legible to someone arriving from the citation: what to run, in
what order, what each output corresponds to in the paper, and expected
runtimes. Include the figure manifest mapping each file to its manuscript
location.
