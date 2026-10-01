# START HERE

**This is the only file a new Claude Code window needs to be pointed at.** It
says what to read, in what order, and where the stage's own instructions are.

## The one-line invocation

Paste this into a fresh window:

    Read reports/START_HERE.md and follow it. I am starting Stage 3.

Substitute the stage. Nothing else is needed: `CLAUDE.md` is read automatically
at session start, and this file names everything else.

## What to read, in order

1. **`CLAUDE.md`** -- the project brief, the standing constraints, the pipeline
   roadmap and the decision log. Loaded automatically; read the roadmap row for
   your stage and the last ten decisions before doing anything.
2. **This file, below** -- how the workflow runs and what the standing
   conventions are.
3. **Your stage's section of `reports/STAGE_PROMPTS.md`** -- the instructions
   for the work itself. The sections are:

        Stage 0    line   578   SENT AND RUN, a record, do not edit
        Stage 1    line   647   SENT AND RUN
        Stage 2    line   756   SENT AND RUN, covers 2a through 2h
        Stage 2j   line  2785   SENT AND RUN 2026-09-25
        Stage 3    line  2914   LIVE
        Stage 4    line  3939   LIVE

   **Stage 2j has been sent and run and its text is now a record too**, so
   only Stage 3 and Stage 4 may be edited. **Stage 2i is closed** and is not
   coming back; the real-building anchor comes from citing Marsh et al. (in
   press). **The configuration block at the top of that file supersedes any
   value quoted inside a sent stage.**
4. **`CONTEXT.md`** -- mechanics: package layout, the fitting interface, how to
   run the pinned environment, the table inventory, the test suite. Read it
   before touching code.
5. **`reports/STAGE_REPORT_<previous stage>.md`** -- what the last stage found,
   what moved, and what it left open.

## What this window produces

**One file: `reports/STAGE_REPORT_<your stage>.md`.** Its specification is in
`CLAUDE.md` under "Stage report specification". It is read by the author and by
a fresh review window, neither of which has done the work.

---

## What changed

This project has been run across two surfaces since Stage 0. Claude Code did the
analysis in the repository; a separate chat window with no repository access
reviewed each stage's handoff, edited the staged prompt file, and held the
context from the author's advisor.

**That split is ending. Claude Code now owns the prompt file as well as the
analysis.** The author is not going to carry findings between two windows by
hand any more.

Two things follow, and the second is the one that is easy to get wrong.

## 1. You own `reports/STAGE_PROMPTS.md`

Edit it directly. The rules on it are unchanged and they are strict:

- **Stages 0 through 2h have been sent and run. Their text is a RECORD and must
  not be edited**, not even to correct a number that has since changed. The file
  is the only account of what each session was actually given. Where a sent stage
  disagrees with current reality, the configuration block at the top of the file
  supersedes it.
- **Only Stage 2j, Stage 3 and Stage 4 are live.** Stage 2i is closed and is not
  coming back; the real-building anchor comes from citing Marsh et al. (in press).
- When you change a prompt, say so in your stage report and move on. Do not
  hand the author a list of things to do that are yours to do.

**The configuration block at the top of that file supersedes every value quoted
inside a sent stage, and keeping it true is part of closing a stage.** It was
rebuilt at the close of Stage 2h after the dump that produced the previous
version raised `KeyError: 'count'` partway through and was pasted with its
traceback unread. If your stage changes a production constant, change it there
in the same commit, and say in your stage report that you did.

## 2. You have to replace the outside reader, deliberately

The chat window's real function was not prompt editing. It was that it had not
run the analysis. Across the last three stage reviews it caught, among other
things: that six audit results were computed on a corpus that had been replaced
later in the same stage; that the generator-parameter sweep's conclusion was
measured against a contaminated objective; that the scorecard figure's caption
still carried nine numbers from the superseded corpus while the figure itself was
current; and that two sections of one handoff contradicted each other about
whether a sourcing gap was open.

None of those needed repository access. All of them needed a reader who did not
already believe the session's own account of what it had done. **A session that
has just done the work does not interrogate the assumptions the work was built
on, and that is a property of context rather than of capability.**

**So keep the separation and move it inside Claude Code.** At the close of every
stage, before the next stage runs:

1. The stage window writes `reports/STAGE_REPORT_<id>.md` to the specification
   in `CLAUDE.md`: every number stated in full as text, every headline claim
   carrying the command that reproduces it and a plain-language "so what",
   every result stating which corpus and which weight rule it ran on, and the
   figures embedded. The older `HANDOFF_stage-*.md` files keep their names and
   nothing is renamed retroactively.
2. **Open a FRESH window whose only job is to read that report and attack it.**
   Give it the report and the prompt file and nothing else at first. **The
   invocation is below and is complete; nothing has to be remembered and added
   to it.**

        Read reports/STAGE_REPORT_<id>.md and reports/STAGE_PROMPTS.md. Your
        only job is to attack that report: find what is wrong, stale,
        internally contradictory, or asserted without measurement. Form your
        questions from the report before you open the repository. Two rules
        bind you: a finding is admissible ONLY if you give the command that
        produces the number, and you are writing to the author, who has not
        done the work -- open each finding with one plain sentence saying what
        is wrong and what it changes, before any table. Its task is
   to find what is wrong, stale, internally contradictory or asserted without
   measurement, and to write the follow-up questions. It may then open the
   repository to check, but it forms its questions from the report first.
3. Only after that window's questions are answered does the next stage start.

**A FINDING IS ONLY ADMISSIBLE IF IT COMES WITH A COMMAND THAT PRODUCES THE
NUMBER. Added 2026-10-01 after the Stage 2j review.** That review found real
defects -- a bolded claim its own table contradicted, a headline resting on 52
datasets, a number counted twice, a wrong first task for the next stage -- and
the author could not read it. Every finding that mattered resolved to a value
that could be re-run; every finding that was merely rhetorical did not, and the
two sets were indistinguishable in the prose. Requiring the command would have
cut that review to a page, and a page the author could read.

**AND THE REVIEW WRITES TO THE AUTHOR, NOT TO THE SESSION THAT DID THE WORK.**
Give it the report and the tables, and tell it so explicitly. Left alone it
inherits the register of the documents it is given -- which is this repository's
house style, dense and bolded -- and produces something only the window that
wrote the analysis can read. Each finding opens with one plain sentence saying
what is wrong and what it changes, before any table or number.

The review window's standing questions, which this project has learned the hard
way:

- **Which corpus, and which weight rule, did each result run on?** Ask per result,
  with file timestamps, not per stage.
- **Does any number in this handoff appear twice at two values?** Check the prose
  against the tables and the figure captions against both.
- **Does any section contradict a later section?** Handoffs are written across a
  session and the early parts go stale inside the file.
- **Is any conclusion measured against a baseline, a default or a noise level that
  has since changed?** If the thing you compared against moved, the comparison did
  not survive.
- **Did any script in this stage fail, print a traceback, or print a verdict it
  could not have computed?** Read the output, not the summary of it.

## 3. What stays out of the repository

**The manuscript docx with the advisor's 98 unresolved comments must never be
committed and must not be placed inside the repository tree at all, gitignored or
otherwise.** The repository is public and Zenodo-archived, and a gitignored file
is one `git add -f` or one careless `.gitignore` edit away from publishing an
advisor's private comments. If a session needs it, the author supplies an absolute
path outside the tree.

The manuscript revision itself is not Claude Code's work and is not happening
here. It is prose work that needs the advisor's markup and the reference PDFs and
none of the code.

`refs/` is untracked by decision 1, because it holds copyrighted publisher PDFs.

## 4. Standing conventions, carried forward

- **Vocabulary, settled at the close of Stage 2h.** The two weighting schemes are
  **"market weights"** and **"uniform weights"**; the oracle scheme is **"known
  market shares"**. "Variable", "sampled market shares" and "Dirichlet shares" are
  retired and must not appear in an axis label, legend, panel title, column name
  or filename. One sentence travels with the label at first use: market weights
  are DRAWN from a Dirichlet because production volumes are not published, so the
  label does not mean real production volumes.
- **Every summary statistic says whether it is the error in ONE decision or the
  error in an AVERAGE of many.** For five of the study's sixteen headline claims
  those differ by about a factor of twenty, and five rows of the scorecard were
  computed one way while their captions claimed the other.
- **Judge a calibration change against the weight-draw noise of 0.006 to 0.015**,
  not the generator seed noise of 0.0066. The weight draw is the larger of the two
  and six stages quoted the smaller one without knowing.
- **Fitted crossings are fitted twice**, parametric and monotone-nonparametric.
  Tables and supplement carry both at full precision with the bootstrap interval;
  prose and figure annotations round at the first digit where the two fits
  disagree.
- **Plain ASCII, US spelling, in every file this project writes.** No Unicode
  subscripts, no Unicode minus, no Unicode multiplication sign.
- **An absolute distance is not a quality score.** Every goodness-of-fit number
  rose 16 to 45 percent when the corpus was made more dispersed and the fits did
  not get worse. Any check written against an absolute band is suspect.

## 5. Where the work stands

Stages 0 through 2h are run and Stage 2j is run. Stage 2i is closed.
**Stage 3 runs next**, then Stage 4, which is required rather than optional
because the code is cited in the paper as a public Zenodo deposit.

**THE FLIP-THRESHOLD ITEM IS CLOSED and this section said otherwise until
2026-09-25.** `flip.FLIP_THRESHOLDS` was a hard-coded constant calibrated on the
superseded corpus whose three values had all fallen outside their own recomputed
intervals. The author took the recalibration at the close of Stage 2h: the
constants moved from 0.0018, 0.011 and 0.025 to **0.0029, 0.015 and 0.032**, all
three now sit inside their intervals, and notebook 1 was re-run on them, which
moved the mean probability that unknown market shares change which material
leads from 0.9870 to 0.9734 at the 1 percent level, 0.8642 to 0.8144 at 5
percent and 0.7105 to 0.6524 at 10 percent. The control that says the change
reached only what it should: the separation columns, which do not read the
constant, are bit-identical. Nothing is waiting on the author here.
