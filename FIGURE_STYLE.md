# FIGURE_STYLE.md - how every figure in this repository is built

Binding on every figure this repository produces, for the manuscript and for the
supplement alike. It follows Edward Tufte, *The Visual Display of Quantitative
Information* and *Envisioning Information*, and Jean-luc Doumont, *Trees, maps,
and theorems*. Where the two differ in emphasis, Doumont governs the MESSAGE and
Tufte governs the INK.

`src/figstyle.py` implements what can be implemented. Read this file before
writing a figure and call that module rather than re-deriving it.

---

## 1. The one rule that matters most: a figure carries ONE message, and the title states it

Doumont's central claim is that every piece of communication should be built
around a single message the audience can restate afterwards. His own test, from
a worked counterexample: a display conveys no message when its "title expresses
what the data are, not what the data mean -- **the what, not the so what**". A
title that names the variables is a label; a title that states the finding is a
message.

    WRONG   "A_IQR against dataset size"
    RIGHT   "A_IQR measures how many EPDs a category has, not how spread it is"

    WRONG   "Flip probability against relative W1"
    RIGHT   "Below a relative W1 of 0.002 the answer almost never changes"

**If you cannot write the takeaway in a sentence, the figure is not ready and no
amount of styling will save it.** Write the title first, then build the panel
that earns it.

A multi-panel figure has one message overall and one per panel. The overall
message goes in the caption or in a `suptitle`; each panel title carries its own.
If two panels make the same point, delete one. If a panel has no point, delete
it -- this is the commonest fault and the hardest to act on, because the panel
usually took work.

**Do not let a title assert more than the panel shows.** Graphical integrity is
Tufte's first demand: the representation must be proportional to the quantity,
and the words must be proportional to the evidence.

### The description belongs in the CAPTION, not on the figure

**Added 2026-09-27, Stage 2j review, after the author read a figure whose
subtitle ran wider than the figure itself.** Their words: "why do you have a
paragraph of text in this figure that extends way beyond the width of the
figure itself? First of all, that paragraph of text would go in the figure
description, not on the figure itself. Second of all, text breaks should be
used to align with the width of the actual figure."

Three rules, and the first is the one that was being broken:

- **Anything longer than two short lines goes in the CAPTION** -- the
  paragraph beside the figure in the report and in the manuscript -- and not
  on the figure. What stays on the figure is the title, a subtitle of at most
  two lines saying what is plotted, and the axis labels.
- **Every line of title and subtitle is broken by hand to the figure's own
  width.** Matplotlib will not wrap for you and `bbox_inches='tight'` will
  happily widen the saved image to fit a long line, so the text sets the
  figure width instead of the other way round. Put the newline in yourself and
  look at the result.
- **A takeaway title still has to say what the reader is looking at.** The
  same author, same review: "the title used is ambiguous and doesn't clearly
  explain what I'm actually looking at. You're overcorrecting in favor of Jean
  Luc Doumont's rule about having a takeaway as the title." A title that is
  only a conclusion leaves a reader who has not read the text with nothing to
  orient on. Name the comparison in the title, then state the finding.

**And a title may not assert more than the panel shows**, which is section 1's
rule and is easy to break by writing the number into the string by hand.
Compute it: a title that says "beats every method on all sixteen claims"
should be built from the count in the table, so that it reads "on 14 of 16" if
that is what the data says.

### A takeaway title needs a subtitle saying what is plotted

**Added 2026-09-22, Stage 2g, after the author reviewed three figures built to
this file and could not tell what any of them showed.** Their words: the titles
"are so vague, have no description, and leave you completely in the dark about
what's actually shown in the plot", and the axis labels "keep getting drawn out
to 2-3 sentences rather than just a succinct, clear label".

That is a failure of THIS FILE, not of the author's reading. Section 1 says the
title carries the message and says nothing about where the description goes, so
a writer following it puts the message in the title and then has nowhere to say
what is on the axes except the axis label, which then becomes a paragraph.

**Three slots, three jobs, and none of them does two:**

    title       the message. What the panel means. A sentence.
    subtitle    what is plotted. Small, gray, directly under the title.
                One line, occasionally two.
    axis label  a short noun phrase with its units. NOT a sentence.

`CompareUQMethods_FIG_ClaimScorecard` is the worked example: "Under the BEST of
the six methods a probabilistic LCA is right to 1 pct on the design comparison
and wrong by 32 pct on which material leads" as the message, "a black box marks
the method closest to the truth on that row" as the gray subtitle, and "mean
absolute error against the true parent, as a pct of the true level of the same
quantity" as the colorbar label.

**Both numbers in that title are COMPUTED from the table the figure draws**,
never typed. A title with a hardcoded number drifts away from its own panel on
the next run, and nothing catches it.

An earlier worked example here was `CompareUQMethods_FIG_TailBlindSpot`, whose
title was "W1 stops charging once the tail leaves its grid". It was cut, and it
is worth recording why the title failed: "charging" was internal shorthand for
"adding to the W1 score". **A title that needs its own vocabulary explained is
not a title.** Write the message in the words a reader of the paper already
has.

**The subtitle is also where a series legend belongs when direct labeling will
not fit.** Naming two lines in a subtitle costs one short line; labeling them
at their ends cost a collision with the panel next door.

### A takeaway title is not a licence to hide the data

The same review: "Why are we showing a range without labeling the UQ methods? We
have a color coding system for which UQ method is which, what's the purpose of
hiding that?" A figure that aggregates away the thing a reader came for is not
saved by a good title. **If the panel has room to show all the levels of a
factor, show them.** The figure that prompted this was cut rather than fixed.

---

## 2. Data-ink: erase, then erase again

Tufte's test is the ratio of ink that encodes data to ink on the page. Every mark
that is not data must justify itself.

Tufte's own list is five instructions: above all else show the data; maximize
the data-ink ratio; erase non-data-ink; erase redundant data-ink; revise and
edit. The last is the one people skip.

**Erase by default:**

- the top and right spines, always;
- gridlines, unless a reader genuinely needs to read values off the axis, in
  which case use one faint set on one axis only;
- boxes around legends, around panels, around anything;
- background fills and shading of any kind;
- tick marks that duplicate a labeled value;
- minor ticks, unless a log axis genuinely needs them;
- redundant axis labels on shared axes in a small-multiple grid;
- any use of color that also has a position or a shape encoding the same thing.

**Keep and strengthen:**

- the data marks themselves, which is Tufte's first principle: above all else
  show the data;
- one direct label per series, placed at the series, replacing the legend;
- the smallest number of axis ticks that lets a reader interpolate, typically
  three to five;
- annotation of the specific points the message depends on.

**Direct labeling beats a legend.** A legend forces the reader to look away,
decode a color, and look back. Doumont's objection is blunter: a legend is "an
arbitrary dictionary of colors, hard to process". Put the series name at the end of the series, in the series
color. Use a legend only when lines are too dense to label in place.

---

## 3. Small multiples over multi-series clutter

When a relationship should be compared across a third variable, repeat the panel
rather than overplotting. **Hold the axes identical across panels**, which
Doumont states as a requirement rather than a preference -- panels showing
subsets of the same variables "must use the same scales to offer a meaningful
comparison" -- so the comparison is positional, label the axes once, and let the panel titles carry the level of
the third variable. Small multiples are Tufte's strongest recommendation and this
project's data -- two arms, four size bands, six UQ methods -- suits them.

---

## 4. Color

Color encodes, it does not decorate.

- **Categorical:** at most six, from a colorblind-safe set. `figstyle.CATEGORICAL`
  is Okabe-Ito, which is safe for deuteranopia and protanopia and prints legibly
  in grayscale.
- **Sequential:** one perceptually uniform ramp, `viridis` or `cividis`. Never
  `jet`, `rainbow` or any ramp with a luminance reversal, which manufactures
  boundaries that are not in the data.
- **Diverging:** only where the variable genuinely has a meaningful midpoint, and
  then anchored at that midpoint.
- **Grayscale first.** If the figure fails in grayscale, position, shape or
  direct labeling is doing too little work.
- **One accent.** Reserve a single saturated color for the thing the message is
  about, and render everything else in gray. A figure where everything is
  colored emphasizes nothing.
- **One color scale means ONE quantity.** Added 2026-09-23, Stage 2g, after a
  heatmap drew seventeen rows on a shared ramp where seven of them were divided
  by one thing and ten by another. Both were percentages, so nothing on the page
  said they were different statistics. **If the cells of a shared scale are not
  computed the same way, the scale is a lie, and naming the denominators in the
  row headers does not repair it** -- the reader still has to do arithmetic the
  color has already done wrongly. Either put every cell on one definition or
  use separate panels with separate scales.

---

## 5. Text on figures

- Plain ASCII only, as everywhere in this project. No Unicode minus, no
  multiplication sign, no typographic quotes. Write `CO2`, not a subscript.
- One font family throughout; the default sans is fine.
- Sizes: panel title 9 pt, axis label 8 pt, tick label 7 pt, annotation 6.5 pt.
  Set them through `figstyle.apply()` rather than per call.
- **Annotations must not overlap data or each other.** If a label collides,
  move the label, not the data, and if there is nowhere to move it the panel is
  too crowded. Check every figure at final size before committing it; a label
  that is legible at 200 percent zoom and collides at 100 percent is a defect.
- State units once, in the axis label, never on every tick.
- Numbers on a figure carry the precision the estimate supports and no more. A
  crossing whose bootstrap interval is 30 percent wide is quoted to two
  significant figures.

---

## 6. Axes

- Log scales wherever a quantity spans more than about two orders of magnitude,
  which in this project is most of them. Say so in the axis label if it is not
  obvious from the ticks.
- Do not start a bar chart's axis anywhere but zero. For point and line charts,
  a non-zero baseline is acceptable and often necessary, but the range must not
  be chosen to exaggerate a difference: Tufte's lie factor should be near one.
- Identical scales across panels that are meant to be compared. Different scales
  across panels that are not, with the difference made obvious.
- No secondary y-axis. It invites false inference about crossings that are an
  artifact of two arbitrary scalings. Use two panels.

---

## 7. Mechanics in this repository

- **Figures are generated from tables on disk, never from in-memory state.** A
  standing project rule, and it is what makes a figure reproducible without a
  full notebook run.
- **`outputs/` is written by the notebooks and by nothing else.** Audit scripts
  may write only under `outputs/tables/audits/`.
- Save at `dpi=300` with `bbox_inches='tight'`. Never set `figure.dpi` high and
  rely on `savefig.dpi` defaulting to `'figure'`; that silently produced enormous
  files in this project once already.
- Name files `CompareUQMethods_FIG_*.png` for the manuscript and
  `CompareUQMethods_SUPP_*.png` for the supplement.
- Width: a single-column figure is about 3.5 in, a full-width figure about 7.2
  in. Build at final size. A figure designed at 11 in and shrunk to 7 has
  illegible text, which is the most common way this rule is broken.

---

## 8. The checklist, before any figure is committed

1. Can I state the takeaway in one sentence? Is that sentence the title?
2. Does every panel earn its place, and does each make a different point?
3. Have I deleted the top and right spines, the gridlines, the legend box and
   every background fill?
4. Is each series labeled where it is drawn, rather than in a legend?
5. Does it survive grayscale?
6. Is the color ramp perceptually uniform, and is the accent color on the
   thing the message is about?
7. Do any two pieces of text touch, at final size?
8. Is it built at final width, with text legible there?
9. Is every number quoted to a precision the estimate supports?
10. Does the figure read without the caption? Does the caption add something
    rather than repeat the title?
11. **Doumont's test:** show it to someone representative of the audience with
    no spoken explanation. Can they say what it shows and why it is there? If a
    reader has to ask what a mark means, the mark is not labeled.

---

## 9. What comes from where

**Sourced to Tufte**, *The Visual Display of Quantitative Information* (1983):

- the five data-ink principles, quoted -- **above all else show the data;
  maximize the data-ink ratio; erase non-data-ink; erase redundant data-ink;
  revise and edit**;
- the data-ink ratio itself, defined as one minus the proportion of the graphic
  that could be erased without loss of data information;
- **chartjunk**, and its three named forms: moire vibration, heavy grids, and
  self-promoting graphics;
- the **lie factor**, the size of the effect shown divided by the size of the
  effect in the data, which should be near one;
- **small multiples**, which is his term.

**Sourced to Doumont**, *Trees, maps, and theorems* (2009):

- the three laws -- **adapt to your audience, maximize the signal-to-noise
  ratio, use effective redundancy**;
- the message rule, in his words: a display that conveys no message is one whose
  "title expresses **what the data are, not what the data mean (the what, not
  the so what)**";
- the objection to legends, from his worked counterexample: a legend is "an
  arbitrary dictionary of colors, hard to process";
- the small-multiple constraint: "multiple panels representing subsets of the
  same variables **must use the same scales** to offer a meaningful comparison";
- the self-explanation test -- show the display to someone representative of the
  audience, without your spoken text, and see whether they can say what it shows
  and why it is there.

**This project's own conventions, which neither author states.** The Okabe-Ito
palette and the six-color limit; the specific point sizes; the ASCII-only rule,
which is a standing constraint of this repository and not a design principle;
the file-naming scheme; the 3.5 and 7.2 inch widths; and the requirement that
figures be built from tables on disk. These are ours. They are listed separately
so a future stage can change them without arguing with Tufte.

---

## 10. Provenance of this file

Written 2026-09-17, Stage 2d, at the author's instruction, after a review found
that the stage's three figures followed no written guide: titles named variables
rather than findings, annotations collided with data, and legends were used where
direct labels would serve.

**Revised the same day**, also at the author's instruction, after checking the
primary sources rather than working from memory. The first draft attributed to
Tufte and Doumont several rules neither of them states; section 9 now separates
what is sourced from what is ours.

**Revised 2026-09-22, Stage 2g**, after the author reviewed three figures built
to this file and found the titles uninformative and the axis labels running to
sentences. Section 1 gained the three-slot rule -- title, subtitle, axis label --
and the note that a takeaway title does not license hiding the data. One figure
was cut rather than repaired.

**Revised again 2026-09-23, same stage.** A second figure was cut, its title
having needed a glossary. Section 4 gained the rule that one color scale means
one quantity, and section 1's worked example moved to the figure that survived
and gained the requirement that a number in a title be computed from the table
rather than typed.

**Before this file existed the guide lived only in conversation, which is the
condition this project's own continuity rule forbids.** If a figure convention is
decided in future and is not written here, it does not exist.
