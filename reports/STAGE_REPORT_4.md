# Stage 4: the deposit

**Corpus `corpus_2026-09-25`. Empirical weight rule `WEIGHT_RHO = 0.5`.** Every
number below ran on those two and nothing else. Branch `stage-4-deposit`, from
`a28d322`. The working tree was already clean, so nothing needed committing
first.

---

## The first page

**Four code changes, then one run of notebook 3, then the controls -- the order
the Stage 3 review set (decision 239).** All four are in. The run is reported
in sections 1 to 3.

**PLACEHOLDER: headline counts after the run.**

**What needs a decision: nothing.** The figure numbering, full
`FIGURE_STYLE.md` compliance and the confidence intervals on figure aggregates
all wait on the manuscript's figure selection, by decision 235, and this stage
did not touch them.

**One thing the Stage 3 review got wrong, and it is worth saying because it
would otherwise be done twice.** Decision 241 asks this stage to add (a) (b)
(c) (d) panel labels to the pLCA scatter figure. **They have been there since
2026-05-12** -- `git log -S "alphabet[ires]"` -- and the committed PNG shows
them. What is actually wrong with that figure is that the top row's x axis
labels print on top of the bottom row's titles, which is visible in the
committed image and was not in the review. That is fixed instead.

---
