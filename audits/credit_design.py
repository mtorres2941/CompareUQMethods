"""The certification credit as a whole-design claim. Stage 2h.

THE AUTHOR'S FRAMING. Practitioners run LCAs to earn green-building points, and
the credit is a threshold: demonstrate a reduction of at least X against a
baseline. Under a probabilistic LCA that becomes a threshold on a CONFIDENCE --
"demonstrate a 10 percent reduction with 75 percent confidence" -- and the
question is whether the choice of UQ method changes whether you earn it.

A CREDIT IS A WHOLE-DESIGN CLAIM AND NOT A ONE-MATERIAL ONE, which is why this
uses the design swap rather than the reduction strategies. Capping a single
material almost never moves a whole building by 10 percent, so asking the
credit question of a single-material intervention answers it in the regime
where nobody is near the bar. `audits/credit_threshold.py` does that and its
own output shows the problem: at a 10 percent tier and 75 percent confidence
the true distributions clear the bar in 0.24 percent of cases.

AND THE EXISTING MARGIN GOES THE WRONG WAY FOR THIS QUESTION.
`plca.comparison_statement` computes `P(a < g * b)`, and the study's margins are
1.0, 1.05 and 1.2. With `a` the proposed design and `b` the baseline, a margin
ABOVE one asks "is the proposal better, OR worse by less than g", which is a
tolerance. A credit asks the opposite: "is the proposal better BY AT LEAST a
margin", which is `g` BELOW one. Verified on the study's own run: at a true 20
percent saving `mci_1.2` reads 0.9993 against a discernibility of 0.9628, so it
is the looser condition and not the stricter one. **The docstring on that
function says "the share in which A beats B by a margin worth acting on", which
describes `g < 1` and not what the code computes; see the handoff.**

So this runs the design swap again with credit margins, and asks of each pair:

    does this method say the proposal beats the baseline by at least the tier,
    with at least the required confidence -- and does the truth agree?

    conda run -n compareuq python audits/credit_design.py [n_pairs]
"""
import os
import sys
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import corpus                # noqa: E402
import fitting as FT         # noqa: E402
import plca as PL            # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'audits')

#: Credit tiers, as the fraction the proposal must beat the baseline by.
#: `g = 1 - tier`, so a 10 percent credit is `P(proposal < 0.90 * baseline)`.
TIERS = (0.05, 0.10, 0.20)

#: Confidence levels a credit might be written at. 0.5 is "more likely than
#: not", which is what a deterministic LCA implicitly claims when it reports a
#: point estimate on the right side of the line.
CONFIDENCES = (0.50, 0.60, 0.75, 0.90, 0.95)

#: True expected savings to build designs at, so the sweep covers designs that
#: genuinely clear each tier, designs that genuinely miss it, and designs
#: sitting on it. Without the last group the disagreement rate is near zero by
#: construction and says nothing.
TRUE_SAVINGS = (0.0, 0.05, 0.10, 0.15, 0.20)

CREDIT_MARGINS = tuple(round(1.0 - t, 4) for t in TIERS)


def main(n_pairs=600, neccs=4000, nmats=4):
    os.makedirs(TABLES, exist_ok=True)
    pd.set_option('display.width', 220)
    rng = np.random.default_rng(20260924)

    met, values, _ = corpus.load_corpus(with_values=True)
    met = met[met.dataset.astype(str).str.startswith('dataset')]
    groups = PL.resample_groups(sorted(met.dataset.astype(str)), nmats + 1,
                                n_pairs, rng)
    needed = sorted({d for g in groups for d in g})
    vals = corpus.as_dict(values)
    samplers = PL.LazySamplers(corpus.load_parent_objects(datasets=needed),
                               scheme=PL.TRUTH_SCHEME)
    models = {}
    for d in needed:
        x, w = np.asarray(vals[d][0]), np.asarray(vals[d][1])
        models[d], _ = FT.fit_pewt(x, w)
    print(f'{n_pairs} design pairs, {len(needed)} datasets, '
          f'{neccs} draws each', flush=True)

    df = PL.swap_run(models, groups, rng, savings=TRUE_SAVINGS, neccs=neccs,
                     samplers=samplers, margins=CREDIT_MARGINS)
    df.to_csv(os.path.join(TABLES, 'TABLE_CreditDesignSwap.csv.gz'),
              index=False)
    report(df)
    return df


def report(df):
    rows, cliffs = [], []
    for tier, g in zip(TIERS, CREDIT_MARGINS):
        col = f'mci_{g:g}'
        if col not in df.columns:
            continue
        for conf in CONFIDENCES:
            w = df[['pair', 'saving', 'method', col, f'{col}__truth']].dropna()
            w = w.rename(columns={col: 'p', f'{col}__truth': 'p_truth'})
            w['earned'] = w.p >= conf
            w['earned_truth'] = w.p_truth >= conf
            per = w.groupby(['pair', 'saving'])
            disagree = per.earned.nunique() > 1
            wrong = w.assign(bad=w.earned != w.earned_truth).groupby(
                'method').bad.mean()
            rows.append(dict(
                tier=tier, confidence=conf, n_cases=int(per.ngroups),
                truth_earns=float(per.earned_truth.first().mean()),
                methods_disagree=float(disagree.mean()),
                best_method_wrong=float(wrong.min()),
                worst_method_wrong=float(wrong.max()),
                worst_method=str(wrong.idxmax())))
            if tier == 0.10 and conf == 0.75:
                p = per.agg(disagree=('earned', lambda s: s.nunique() > 1),
                            p_truth=('p_truth', 'first'))
                p['d'] = (p.p_truth - conf).abs()
                lo = 0.0
                for hi in (0.05, 0.10, 0.25, 1.01):
                    pick = p[(p.d >= lo) & (p.d < hi)]
                    if len(pick):
                        cliffs.append(dict(from_line=f'{lo:.2f}-{hi:.2f}',
                                           n=len(pick),
                                           disagree=float(pick.disagree.mean())))
                    lo = hi
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TABLES, 'TABLE_CreditDesign.csv'), index=False)
    print()
    print('=' * 78)
    print('DOES THE UQ METHOD CHANGE WHETHER THE CREDIT IS EARNED?')
    print('A credit of "beat the baseline by TIER with CONFIDENCE confidence".')
    print('=' * 78)
    print(out.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print()
    print('BY HOW MUCH THE DESIGN ACTUALLY BEATS THE BASELINE, at a 10 pct')
    print('credit and 75 pct confidence -- which says whether the fragility is')
    print('the threshold or the methods:')
    g = CREDIT_MARGINS[TIERS.index(0.10)]
    col = f'mci_{g:g}'
    if col in df.columns:
        w = df[['pair', 'saving', 'method', col, f'{col}__truth']].dropna()
        w['earned'] = w[col] >= 0.75
        t = w.groupby(['saving', 'pair']).earned.nunique() > 1
        tt = (w.groupby(['saving', 'pair'])[f'{col}__truth'].first() >= 0.75)
        print(pd.DataFrame({'methods_disagree': t.groupby('saving').mean(),
                            'truth_earns': tt.groupby('saving').mean()})
              .to_string(float_format=lambda v: f'{v:.4f}'))
    if cliffs:
        print()
        print('AND BY HOW FAR THE TRUE CONFIDENCE SITS FROM THE LINE:')
        print(pd.DataFrame(cliffs).to_string(index=False,
                                             float_format=lambda v: f'{v:.4f}'))


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 600)
