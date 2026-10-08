# Review of the figure pass

A fresh-window review of `reports/REPORT_FIGURE_PASS.md`, of narrative sections 4
and 9, and of `git diff 4055ccc -- reports/MANUSCRIPT_NARRATIVE.md`, written
2026-10-08 for the author. Everything runs on `corpus_2026-09-25` at
`weight_rho = 0.5`. Every command runs from the repository root under the
`compareuq` environment.

Seven findings, most serious first, then what was checked and holds.

---

## 1. Figure 5's shaded band belongs to a rule nobody can use, uses the tie test decision 224 retired, and prints numbers decision 225 forbids

**What is wrong.** Figure 5 shades 68 to 106 EPDs and labels it "best fit". That
band is the cutoff for "kernel estimate with MARKET weights above, lognormal with
uniform weights below", which is the known-share rule decision 216 set aside
because it needs market shares. It was also computed with the old tie test,
which compares each cutoff against whichever cutoff won that resample. Decision
224 replaced that test because it is a contest among neighboring grid points, not
a difference test. And decision 225 says the paper prints two ranges only, 40 to
170 and 80 to 100. Figure 5 prints three more: 68-106, 62-88 and 40-170.
**What it changes:** the Figure 5 caption, takeaway 8, and the qualification
section 9 hands to the prose window ("68 to 106 is the band of fit-level CUTOFFS
indistinguishable from the best") all describe the wrong rule, measured by a
retired method.

Rerun at fit level with the decision-224 test (a fixed reference: the cutoff
that wins on the full sample):

    rule                                       argmin   fixed-reference band   old contest band
    known-share (what Figure 5 shades)           81        68 to 126             68 to 106
    feasible: uniform weights throughout        150        97 to 196             97 to 179

At fit level, the rule a reader can follow is best between about 97 and 196 EPDs,
not 68 to 106. The table has no corpus stamp
(`TABLE_ReductionThreshold.csv`; decision 202 requires one). The cell's own
comment says the families cross at "48 to 73", while the figure prints 62 to 88
(`metricreduction.crossover_band` reproduces 62 to 88).

    python -c "
    import pandas as pd, numpy as np
    s=pd.read_csv('outputs/tables/TABLE_MethodScores.csv'); s=s[(s.arm=='synthetic')&s.w1_market.notna()]
    w=s.pivot_table(index='dataset',columns='method',values='w1_market').dropna()
    n=s.drop_duplicates('dataset').set_index('dataset').n.reindex(w.index).to_numpy(float)
    T=np.unique(np.round(10**np.linspace(0,3.4,90)).astype(int)); o=w.min(axis=1).to_numpy()
    for hi,lo in [('KDE, Variable','Lognormal, Uniform'),('KDE, Uniform','Lognormal, Uniform')]:
        r=(np.where(n[None,:]>=T[:,None],w[hi].to_numpy()[None,:],w[lo].to_numpy()[None,:])-o)/o; p=int(np.argmin(r.mean(1)))
        g=np.random.default_rng(1); D=np.empty((2000,len(T))); C=np.empty_like(D)
        for b in range(2000):
            i=g.integers(0,len(n),len(n)); c=r[:,i].mean(1); D[b]=c-c[p]; C[b]=c-c.min()
        f=T[np.percentile(D,2.5,0)<=0]; k=T[np.percentile(C,2.5,0)<=0]
        print(hi,'argmin',T[p],'fixed',f.min(),f.max(),'contest',k.min(),k.max())"

**Decision for you.** Remove the band and its numbers from Figure 5, or shade the
feasible rule's band. Either way, takeaway 8 and section 9 item 1's third bullet
need rewriting.

## 2. Figure 8 needs its benchmark, and F4's reason for giving it none does not hold

**What is wrong.** F4 says Figure 8 needs no interval because it is "a count over
the whole real arm". But each point is a skewness estimated from as few as ten
EPDs, and sample skewness runs low for skewed data. Simulate a TRUE
two-parameter lognormal for each of the 127 categories, at that category's size
and coefficient of variation: only about 44 of 127 land within 25 percent of the
curve, and the median ratio is 0.66. **What it changes:** "only 27 of 127 sit
near the curve" and "0.58 times as skewed" are partly what perfect lognormal data
would show anyway. The real categories ARE further from the curve than lognormal
data would be (27 against 34 to 55; 0.58 against 0.61 to 0.74), but by much less
than the figure suggests. The real arm is also trimmed by the 3 x IQR rule,
which lowers skewness further, so if anything the gap is overstated.

    real categories              27 of 127 within 25%   median ratio 0.576
    true 2-param lognormals      44 of 127 (34 to 55)   median ratio 0.661 (0.607 to 0.744)
    (200 simulated replicates, same n and CV per category, no trimming)

    python -c "
    import sys; sys.path.insert(0,'src'); import numpy as np, pandas as pd, customstats as cs
    e=pd.read_excel('outputs/tables/TABLE_EmpiricalECCMetrics.xlsx')[['n','coeffvar_uw','skewness_uw']].replace([np.inf,-np.inf],np.nan).dropna(); e=e[e.n>=10]
    g=np.random.default_rng(0); on=[]; md=[]
    for _ in range(200):
        rr=[]
        for n,cv in zip(e.n.astype(int),e.coeffvar_uw):
            x=g.lognormal(0,np.sqrt(np.log(cv**2+1)),n); c=x.std()/x.mean()
            rr.append(cs.weighted_skew(x,np.full(n,1/n))/(c**3+3*c))
        rr=np.array(rr); on.append(((rr>.75)&(rr<1.25)).sum()); md.append(np.median(rr))
    print(np.median(on),min(on),max(on),np.median(md),min(md),max(md))"

The algebra (skewness = CV^3 + 3 CV) is unaffected, and so is the argument that a
two-parameter lognormal has no free shape. What has to change is the quantitative
claim, which should be stated against this benchmark. Separately, the figure's
title says "A lognormal cannot match the spread and the skew", but the study's
own lognormal is the three-parameter fit, which can. The title should say
"two-parameter".

## 3. Figure 9's intervals measure how many Monte Carlo iterations were run, not how sure anyone can be that rebar leads

**What is wrong.** The bars in Figure 9 come from resampling the 10,000
iterations while the fitted models stay fixed. That noise falls as more
iterations are run. Under the uniform-weighted lognormal, rebar leads by 0.49
points with a half-width of about 3.4 points, so roughly 480,000 iterations would
make that lead "clear". The larger uncertainty, from fitting a model to 204 EPDs,
is not measured at all. **What it changes:** F2's "so what" ("the methods that
flatten the tail cannot tell rebar from concrete") and section 9's "within
noise" read as if those methods are unsure. They are not. Their point estimates
put the two materials within 0.4 to 2.2 points of each other, and the interval
only says the run was too short to order them. The honest statement is the point
gap itself, plus one sentence that model-fit uncertainty is not shown.

    policy               leader          lead (points)   95% interval
    Normal, uniform      ready-mix 5000      0.43        [-1.7, +2.8]
    Normal, market       rebar               2.23        [-0.1, +4.5]
    Lognormal, uniform   rebar               0.49        [-2.9, +3.9]

    python -c "
    import pandas as pd; d=pd.read_csv('outputs/tables/TABLE_Building138UIInterval.csv')
    p=d.pivot(index='dataset',columns='method',values='ui')
    for m in p: s=p[m].sort_values(); print(m, s.index[-1], round(100*(s.iloc[-1]-s.iloc[-2]),2))
    print(d[d.is_leader][['method','lead_lo','lead_hi']])"

## 4. Figure 3's boxes mean different things in different places, and the legend covers only three of four styles

**What is wrong.** In panel (a) a solid box means "best of the four a reader can
choose". In panel (c) the solid box marks the best of all six methods, so it
sits on market-weighted cells (12.2 and 9.3 for KDE with market weights at 100+
EPDs), which a reader cannot choose. Panel (a) also uses a fourth style, a thin
dotted orange box for "within noise of the dashed box" (for example 7.7, 4.4 and
17.9 in the right block). That style is in neither the legend nor the caption.
The caption's "its row's box" does not say which box, and it says nothing about
the boxes in (c). SUPP7 has the same problems. **What it changes:** a reader who
learns the legend from panel (a) misreads panel (c) as saying market weights are
an available choice.

    python -c "
    import pandas as pd; d=pd.read_csv('outputs/tables/TABLE_ClaimScorecardIntervals.csv')
    b=d[(d.panel=='bands')&(d.stat=='median')]; print(b.groupby('row').apply(lambda g: g.loc[g.point.idxmin(),'method']))"

## 5. Figure 5's subtitle says the opposite of its caption about what each method is scored against

**What is wrong.** The figure's subtitle reads "closest to its known parent",
which describes `w1_parent` (each method against the parent it is estimating).
The curves actually use `w1_market`, all six against the market-weighted parent,
as the caption correctly says. Decision 65 treats the two as different quantities
that cannot be swapped. **What it changes:** one line of figure text.
`notebooks/04_CompareUQ_ReduceMetrics.ipynb` cell 32 draws the subtitle, and
`best_method_curve(..., value='w1_market')` in cell 13 is the curve.

    grep -n "closest to its known parent\|value='w1_market'" notebooks/04_CompareUQ_ReduceMetrics.ipynb

## 6. Colors still collide across figures, and the graphical abstract mixes two Monte Carlo passes

**What is wrong, three small things.**
- The graphical abstract draws the three UNIFORM-weighted fits in the DARK
  shades (`FAMILY_COLORS`). In Figures 1 and 9, dark shades mean market weights.
- Figure 2 draws "synthetic dataset" in `tab:blue` (#1f77b4). That is visually
  identical to Normal-with-market-weights (#1f78b4) in Figure 1 on the facing
  page.
- The abstract's four left bars come from the main truth pass, and its 13.0 bar
  from the separate mixed-policy pass. Decision 237 rebuilt the scorecard on one
  pass for exactly this reason. On one pass the kernel estimate prints 17.6, not
  17.5.

**What it changes:** colors in two figures, and one decimal in the abstract.

    python -c "
    import pandas as pd
    a=pd.read_csv('outputs/tables/TABLE_ClaimScorecardWithRule.csv').groupby('method').median_error.mean()*100
    b=pd.read_csv('outputs/tables/TABLE_MixedPolicyScorecard.csv').groupby('method').median_error.mean()*100
    print(a.round(3).to_dict()); print(b.round(3).to_dict())"

## 7. One comparison is quoted at two values, and the report contradicts itself in three places

**What is wrong.** Takeaway 3 prints the rule against a kernel estimate as 17.4
against 17.5. That is the main pass at the constant 80, a gap of 0.16 points
[+0.01, +0.32]. The next sentence quotes "0.2 points, 0.03 to 0.35", which is the
mixed pass at the best cutoff. The report (F1) notices this and keeps both;
section 9 does not pass it to the prose window. Within the report itself:
- Section 2 says Figure 3 changed "Nothing else" and that SUPP7 is
  "byte-identical". Both stopped being true in the third round. Line 38 admits
  this, but the before/after list you are asked to approve still says it.
- Section 4 says no pre-existing table changed except a gzip header; section 6
  correctly lists `TABLE_GenerationExampleCurves.csv.gz`.
- "S1" in report line 32 and in the notebook 3 comments is now SUPP7.

**What it changes:** pick one pass for takeaway 3, and correct the report's own
change list before you approve against it.

    python -c "
    import pandas as pd; d=pd.read_csv('outputs/tables/TABLE_ClaimScorecardIntervals.csv')
    print(d[(d.panel=='pooled')&(d.stat=='median')][['method','point','diff_lo_four','diff_hi_four']])"
    git diff --name-status 4055ccc -- outputs/tables

---

## Checked and holds

- **Check 3.** Only the expected tables changed against 4055ccc: three new ones,
  plus `TABLE_GenerationExampleCurves.csv.gz`. `TABLE_GenerationExampleValues`
  is untouched. Command: `git diff --name-status 4055ccc -- outputs/tables`.
- **Check 2.** All three interval cells resample the right unit: pLCA group or
  design pair; datasets; Monte Carlo iterations. All three test against a fixed
  full-sample reference. F1's counts reproduce: the rule is best of four on 7
  claims, with 12 four-way ties and 6 seven-way ties. F3's band medians reproduce
  from `TABLE_WeightingBySizeIntervals.csv`. Finding 1 above concerns a band
  this pass did NOT recompute.
- **Check 4.** All three supplement corrections are real.
  `customstats.SILVERMAN_MIN_NEFF = 20.0`, and the archived SUPP5 drew 30.
  `comparison.bandwidth_comparison` scores against the fitted data, so W1 there
  is in-sample. The archived SUPP11 title said "one percent", and the 5% crossing
  is 0.0150.
- **Check 6.** Outside archive/ and the dated records, the only leftovers are a
  docstring in `audits/corpus_examples.py` line 4
  (`SUPP_DatasetExamplesByStratum`) and historical prose in `CONTEXT.md` line
  1378.
- **Check 7.** No "declarations" or "pct" appears in any string a numbered
  figure draws; all the hits are `print` calls. The two exceptions to the
  method colors are in finding 6.
- **Check 1.** Apart from the points above, every caption matches its PNG.
  Figure 2's ring count is 15 (3+6+1+3+1+1).
- **Outside this pass's scope, noticed while checking Figure 2.** The bullet
  beside it says only "two extremes" (Aggregates and the large ready-mix
  classes) fall outside the cloud. `TABLE_CoverageFigureStats.csv` also lists
  ConcreteAdmixtures, DampproofingAndWaterproofing, PowerCabling,
  ProcessedNonInsulatingGlassPanes, WallFinishes and DemountablePartitionTrack.
