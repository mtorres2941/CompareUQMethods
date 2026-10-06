# WRITING_STYLE.md

**Binding on every word of the manuscript, the way `FIGURE_STYLE.md` is binding
on every figure.** Written 2026-10-05 from two sources: the 97 comments and 463
insertions Wil V. Srubar III left on
`CompareUQMethods_BE1_Manuscript_v1_wvs.docx` between 2026-08-21 and 2026-08-31,
and the consolidated principles the author supplied from an earlier analysis of
the same markup.

**The covering instruction, in the advisor's words:** virtually no one will be an
expert in all of the content, so be systematic, break it down, and teach where
needed -- which is mostly everywhere. When in doubt, opt for another sentence to
explain what you mean.

**And the test he set:** he stopped editing partway through and said he wants to
see how the next version responds before reading more. **The unreviewed final
third is therefore the real test**, because it is the part no reviewer has
already repaired.

---

## 1. The nine principles

1. **One claim per paragraph, stated in the topic sentence, defended, then
   cashed out with a consequence.** This is the most repeated note in the
   markup -- comments 528, 541, 592, 645, 762 and 766 are all versions of it.
2. **Teach the reader rather than report to them.** Sequence ideas so the reader
   is ready for each result. Comment 829: "the reader does not yet know what
   Rank #1 Frequency. Reread with new eyes and edit (i.e., teach) accordingly."
3. **Say it plainly; cut what does not serve the claim.** Comments 738, 759,
   764 and 767: "a bit thin -- so what? even needed?"
4. **Engage prior literature on what it FOUND, not on the approach it took.**
   The draft has four places reading "X did Y" with a bracketed note asking what
   they found.
5. **Foreground the contribution.**
6. **Reorganize Results by importance of finding, not by figure-panel order, and
   rebuild the figures to match.** The largest single ask in the markup. Twelve
   paragraphs currently open with a version of "Figure 4e, 4f, 4g and 4h show",
   which is a caption rather than a topic sentence.
7. **Make the generalizability argument explicit and early.** A Table 1 of
   generation parameters, and a demonstration that the synthetic datasets cover
   the empirical ones. Comments 282, 316 and 542.
8. **Define a small controlled vocabulary of named indices and hold to it.** He
   flagged the word "result" four times as too vague to carry meaning.
9. **Figures stand alone**: units, legends, panel labels called out in captions.

**Principles 3 and the covering instruction look like they conflict and do not.**
Cut sentences that do not serve the paragraph's claim; add sentences that explain
it. **Length is not the target; load-bearing-ness is.**

---

## 2. Voice

**"This study" is retired as a grammatical subject.** It appears 27 times in the
draft; "we" appears 3 times in the whole manuscript. That imbalance, not passive
voice as such, is the problem.

- **"We"** when the sentence reports a decision, a judgment, or a claim the
  authors own. *We fitted the threshold by profile likelihood because the global
  maximum likelihood estimate does not exist.*
- **Passive** when the sentence reports a procedure anyone following the method
  would perform identically. *Each dataset was normalized by its own unweighted
  mean.*
- **Never "this study"** as the thing doing something.

---

## 3. The advisor's own habits, to write toward

Drawn from his insertions rather than from his published papers, which are
co-authored and copyedited.

- He signposts explicitly: **"namely"** in place of a colon, numbered enumeration
  inside a sentence, **"i.e."** to gloss a term, **"herein"** to mark scope.
- He **front-loads the claim** and then explains it.
- He likes **"substantiate"** as a reporting verb.
- He converts loose noun phrases into **quoted defined terms**.
- He deletes intensifiers and hedges, then sometimes reinstates them **as a
  specific quantity**.
- He merges and splits sentences rather than trimming words, which is why his
  insertions and deletions roughly balance.

**His editorial voice is an intelligent general reader tracking an argument in
real time, not a specialist checking correctness.** Write for that reader.

---

## 4. The habits to drop, named from the draft

- **Long multi-clause sentences with inline "(i.e., ...)" glosses**, often two or
  three per sentence, pushing the verb far from the subject.
- **Procedural sequencing narrated in the order performed** rather than the order
  the reader needs.
- **Findings organized by figure panel.** See principle 6.
- **Conclusions placed at the end of paragraphs and hedged.** Put the conclusion
  first and state it without the hedge, or state the hedge as a number.
- **Agentless prose.** See section 2.

**What must NOT be sanded off:** the draft is precise, unusually honest about
limitations, and argues well when it lets itself. **The Discussion paragraphs on
the offset heuristic, the flat Dirichlet and the equal-intensity construction are
the best writing in the paper** and already do what the markup asks for. The fix
for Results is largely "write it the way the Discussion is already written."

---

## 5. The controlled vocabulary, which principle 8 requires

Fixed by decision 199 and by the five-question frame. **Nothing outside this list
may be used for these concepts, in text, figures, captions or tables.**

| Use | Never | Why |
|---|---|---|
| **uniform weights** | unweighted, equal weighting | decision 199 |
| **market weights** | variable, Dirichlet shares, sampled market shares | decision 199. Four vocabularies were tried; this is the last |
| **known market shares** | oracle, true weights | the contrast with "market weights" is the point |
| **declarations** (or **EPDs**, consistently, never both) | records, data points, values, ECCs when the count is meant | an EPD is a document, an ECC is the number it carries. The draft and one figure use both for the same count |
| the five questions: **magnitude, attribution, information, action, comparison** | "results", "key results", "takeaways" | "result" was flagged four times as too vague |
| **a probabilistic LCA claim** | a result, an output | principle 8 |

**One sentence must travel with "market weights" at first use**, by decision 199:
they are DRAWN from a Dirichlet because production volumes are not published, so
the label does not mean observed production data. **And on the synthetic arm the
weight on each product group is that group's TRUE share**, so the synthetic
comparison is ignoring a known share against using it, never guessing against
knowing (decision 212).

---

## 6. Numbers in prose

These come from this project's own decisions rather than from the markup, and
they bind the same way.

- **Say whether a statistic is the error in ONE decision or in an AVERAGE of
  many.** For five of the fifteen claims those differ by about a factor of
  twenty (decisions 174, 207).
- **Round a fitted crossing at the first digit where the parametric and monotone
  fits disagree, and print both where they still differ there** (decision 175).
  The tables carry full precision; prose does not.
- **No single-declaration cutoff is printed anywhere.** The family split is 40 to
  170 declarations and the weighting split is 80 to 100 (decision 225).
- **An absolute distance is not a quality score.** Every goodness-of-fit number
  rose 16 to 45 percent when the corpus was made more dispersed, and the fits did
  not get worse.
- **Quote the tables, not the decision log.** The log is the history of how a
  number was arrived at. Eleven places where the two disagreed were corrected on
  2026-10-05 (decision 252) and the log was the stale side in every one.

---

## 7. The test before a section is handed over

Read only the first sentence of every paragraph, in order. **If that sequence is
not a complete and correct argument, the section is not finished** -- which is
principles 1 and 6 applied as a check rather than as advice.
