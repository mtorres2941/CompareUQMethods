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
around a single message the audience can restate afterwards. A title that names
the variables is a label; a title that states the finding is a message.

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

---

## 2. Data-ink: erase, then erase again

Tufte's test is the ratio of ink that encodes data to ink on the page. Every mark
that is not data must justify itself.

**Erase by default:**

- the top and right spines, always;
- gridlines, unless a reader genuinely needs to read values off the axis, in
  which case use one faint set on one axis only;
- boxes around legends, around panels, around anything;
- background fills and shading of any kind;
- tick marks that duplicate a labelled value;
- minor ticks, unless a log axis genuinely needs them;
- redundant axis labels on shared axes in a small-multiple grid;
- any use of colour that also has a position or a shape encoding the same thing.

**Keep and strengthen:**

- the data marks themselves;
- one direct label per series, placed at the series, replacing the legend;
- the smallest number of axis ticks that lets a reader interpolate, typically
  three to five;
- annotation of the specific points the message depends on.

**Direct labelling beats a legend.** A legend forces the reader to look away,
decode a colour, and look back; Tufte calls this an interruption and Doumont
calls it noise. Put the series name at the end of the series, in the series
colour. Use a legend only when lines are too dense to label in place.

---

## 3. Small multiples over multi-series clutter

When a relationship should be compared across a third variable, repeat the panel
rather than overplotting. Hold the axes identical across panels so the comparison
is positional, label the axes once, and let the panel titles carry the level of
the third variable. Small multiples are Tufte's strongest recommendation and this
project's data -- two arms, four size bands, six UQ methods -- suits them.

---

## 4. Colour

Colour encodes, it does not decorate.

- **Categorical:** at most six, from a colourblind-safe set. `figstyle.CATEGORICAL`
  is Okabe-Ito, which is safe for deuteranopia and protanopia and prints legibly
  in greyscale.
- **Sequential:** one perceptually uniform ramp, `viridis` or `cividis`. Never
  `jet`, `rainbow` or any ramp with a luminance reversal, which manufactures
  boundaries that are not in the data.
- **Diverging:** only where the variable genuinely has a meaningful midpoint, and
  then anchored at that midpoint.
- **Greyscale first.** If the figure fails in greyscale, position, shape or
  direct labelling is doing too little work.
- **One accent.** Reserve a single saturated colour for the thing the message is
  about, and render everything else in grey. A figure where everything is
  coloured emphasises nothing.

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
4. Is each series labelled where it is drawn, rather than in a legend?
5. Does it survive greyscale?
6. Is the colour ramp perceptually uniform, and is the accent colour on the
   thing the message is about?
7. Do any two pieces of text touch, at final size?
8. Is it built at final width, with text legible there?
9. Is every number quoted to a precision the estimate supports?
10. Does the figure read without the caption? Does the caption add something
    rather than repeat the title?

---

## 9. Provenance of this file

Written 2026-09-17, Stage 2d, at the author's instruction, after a review found
that the stage's three figures followed no written guide: titles named variables
rather than findings, annotations collided with data, and legends were used where
direct labels would serve.

**Before this file existed the guide lived only in conversation, which is the
condition this project's own continuity rule forbids.** If a figure convention is
decided in future and is not written here, it does not exist.
