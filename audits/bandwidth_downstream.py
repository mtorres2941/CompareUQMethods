"""Which bandwidth rule is best WHERE IT MATTERS: the pLCA answer. Stage 2h.

THE AUTHOR'S CHALLENGE, and it is right. Every comparison of bandwidth rules in
this project has been on a FIT criterion -- leave-one-out likelihood, which is
a density criterion, or W1, which is a CDF criterion. Those two already
disagree (decision 71). But the study exists to say what a probabilistic LCA
reports, and nobody had asked which rule gives the best ANSWER.

WHAT THIS RUNS. The same probabilistic LCA against the TRUE parent
distributions, three times, changing only how much the kernel estimate smooths:

    scott               the rule the manuscript was written against
    silverman           the rule the author's companion paper uses
    silverman_guarded   the study's current rule, Silverman with the scale
                        estimate guarded below 20 effective observations

Both weighting schemes each time, so six kernel variants in all, and the
parametric methods are carried along unchanged as the control: they do not
depend on a bandwidth, so any movement in their columns is noise and bounds
how much of the kernel movement to believe.

    conda run -n compareuq python audits/bandwidth_downstream.py [n_groups]
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

import corpus                 # noqa: E402
import customstats as CS      # noqa: E402
import fitting as FT          # noqa: E402
import metricset as MS        # noqa: E402
import plca as PL             # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'audits')

RULES = ('scott', 'silverman', 'silverman_guarded')

#: The outputs to score. The first two are what a designer reads; the third is
#: the metric this study has historically led with.
OUTPUTS = ('eci_mean', 'eci_std', 'eci_rank_1', 'eci_perc_mean', 'ui')


def fit_with_rule(x, w, rule):
    """The six methods with the kernel estimate at one bandwidth rule.

    The parametric fits do not see the bandwidth, so they are identical across
    rules by construction and serve as the control.
    """
    models, _ = FT.fit_pewt(x, w)
    for scheme, weights in (('Uniform', np.full(len(x), 1.0 / len(x))),
                            ('Variable', w)):
        models[f'KDE, {scheme}'], _ = FT.fit_kde(x, weights, bw_method=rule)
    return models


def main(n_groups=500, neccs=4000, nmats=4):
    os.makedirs(TABLES, exist_ok=True)
    pd.set_option('display.width', 220)
    rng = np.random.default_rng(20260924)
    met, values, _ = corpus.load_corpus(with_values=True)
    met = met[met.dataset.astype(str).str.startswith('dataset')]
    groups = PL.resample_groups(sorted(met.dataset.astype(str)), nmats,
                                n_groups, rng)
    needed = sorted({d for g in groups for d in g})
    vals = corpus.as_dict(values)
    samplers = PL.LazySamplers(corpus.load_parent_objects(datasets=needed),
                               scheme=PL.TRUTH_SCHEME)
    print(f'{n_groups} pLCA groups, {len(needed)} datasets, {neccs} draws',
          flush=True)

    frames = []
    for rule in RULES:
        models = {}
        for d in needed:
            x, w = np.asarray(vals[d][0]), np.asarray(vals[d][1])
            models[d] = fit_with_rule(x, w, rule)
        # THE SAME GROUPS AND THE SAME VARIATES for every rule, so the three
        # are paired and the comparison carries no Monte Carlo noise of its own.
        r = np.random.default_rng(4242)
        df = PL.truth_run(models, samplers, groups, r, neccs=neccs)
        df['bw_rule'] = rule
        frames.append(df)
        print(f'  {rule} done', flush=True)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(os.path.join(TABLES, 'TABLE_BandwidthDownstream.csv.gz'),
               index=False)
    report(out)
    return out


def report(out):
    market = out[out.reference == 'parent'] if 'reference' in out else out
    rows = []
    for rule, block in market.groupby('bw_rule'):
        for m, sub in block.groupby('method'):
            for o in OUTPUTS:
                col = f'{o}__error'
                if col not in sub:
                    continue
                lvl = abs(float(np.nanmean(sub[f'{o}__truth'])))
                if not lvl > 0:
                    continue
                rows.append(dict(bw_rule=rule, method=m, output=o,
                                 rel_error=float(
                                     np.nanmean(np.abs(sub[col])) / lvl)))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(TABLES, 'TABLE_BandwidthDownstreamSummary.csv'),
             index=False)
    print()
    print('=' * 78)
    print('ERROR AGAINST THE TRUE DISTRIBUTIONS, as a pct of the true level')
    print('=' * 78)
    kde = d[d.method.str.startswith('KDE')]
    print('THE KERNEL ESTIMATE, which is what the bandwidth changes:')
    print((100 * kde.pivot_table(index=['method', 'output'], columns='bw_rule',
                                 values='rel_error'))
          .to_string(float_format=lambda v: f'{v:.3f}'))
    print()
    par = d[~d.method.str.startswith('KDE')]
    spread = (par.pivot_table(index=['method', 'output'], columns='bw_rule',
                              values='rel_error'))
    drift = (spread.max(axis=1) - spread.min(axis=1)).max()
    print('THE CONTROL: the parametric methods do not see the bandwidth, so')
    print(f'their largest movement across the three rules is {100*drift:.4f}')
    print('percentage points. Any kernel movement smaller than that is noise.')
    print()
    print('AVERAGED OVER THE FIVE OUTPUTS, per kernel variant:')
    print((100 * kde.pivot_table(index='method', columns='bw_rule',
                                 values='rel_error'))
          .to_string(float_format=lambda v: f'{v:.3f}'))


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
