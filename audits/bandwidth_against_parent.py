"""The bandwidth rule, refereed by the KNOWN PARENT instead of by a proxy.

Stage 2c task 0. `audits/bandwidth_rules.py` chose the guarded Silverman rule on
leave-one-out likelihood, and it had to use a proxy because W1 cannot arbitrate
a bandwidth: in sample W1 falls monotonically as the bandwidth shrinks,
minimizing at about 0.02 of Scott's for most datasets, since a KDE with a
vanishing bandwidth IS the empirical distribution it is being scored against.
Leave-one-out likelihood has no such bias, but it is a different criterion from
the one the study reports, and choosing a method setting on criterion A and
reporting it on criterion B is exactly the objection this stage exists to remove.

THE SYNTHETIC ARM HAS A REFEREE WITH NO BIAS AT ALL. A synthetic dataset was
drawn from a parent this project can write down, so W1 against the PARENT is the
study's own criterion measured against something that is not the training data.
A vanishing bandwidth no longer wins there: it reproduces the sample, and the
sample is not the parent. That is the test the guarded rule was never given, and
this script gives it.

WHAT IT REPORTS. For each rule -- Scott, pure Silverman, the guarded rule in use,
and the bandwidth that MINIMIZES the parent distance, which is the unreachable
optimum -- W1 against the parent, W1 in sample, the leave-one-out referee, and
the fitted model's spread. The optimum column is what says how much any rule
leaves on the table.

IF IT DOES NOT CONFIRM THE GUARDED RULE, SAY SO AND STOP. Changing the guard is
an author decision (decision 54), not something to tune until this script agrees.

    conda run -n compareuq python audits/bandwidth_against_parent.py [n_synth]
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

import corpus  # noqa: E402
import families as F  # noqa: E402
import fitting as FT  # noqa: E402
import mixture as M  # noqa: E402
import recovery as R  # noqa: E402
from bandwidth_rules import loo_loglik  # noqa: E402
from customstats import weighted_std  # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'audits')
SEED = 42
N_SYNTH = 800

#: Bandwidth multiples of Scott's rule swept to locate the parent-optimal
#: bandwidth. Runs well below any standard rule, because that is where the
#: in-sample criterion puts its minimum and the question is whether the parent
#: criterion does too.
SWEEP = np.geomspace(0.02, 3.0, 25)

RULES = ('scott', 'silverman', 'silverman_guarded')


def one(name, x, w, parent, wt, rng):
    ww = FT.uniform_weights(x) if wt == 'Uniform' else np.asarray(w, float)
    ww = ww / ww.sum()
    scheme = R.PARENT_SCHEME[wt]
    sd = weighted_std(x, ww)
    grid = R.recovery_grid(x, w, parent)
    scott = FT.weighted_bw(x, ww, bw_method='scott')

    def at(h):
        m = F.Truncated(F.WeightedKDE(x, ww, h), label='kde')
        return m, R.w1_against_parent(m, parent, scheme, grid)

    swept = [(mult, at(scott * mult)[1]) for mult in SWEEP]
    best_mult = min(swept, key=lambda t: t[1])[0]
    out = []
    named = {r: FT.weighted_bw(x, ww, bw_method=r) for r in RULES}
    named['parent_optimal'] = scott * best_mult
    for rule, h in named.items():
        m, d_parent = at(h)
        out.append(dict(
            dataset=name, n=len(x), weighting=wt, rule=rule, bandwidth=h,
            h_over_sd=h / sd if sd > 0 else np.nan,
            h_over_scott=h / scott if scott > 0 else np.nan,
            w1_parent=d_parent,
            w1_in_sample=FT.score_w1_model(m, x, w),
            loo=loo_loglik(x, ww, h, rng),
            mass_below=m.mass_below,
            best_mult=best_mult))
    for mult, d in swept:
        out.append(dict(dataset=name, n=len(x), weighting=wt,
                        rule=f'sweep_{mult:.4f}', bandwidth=scott * mult,
                        h_over_scott=mult, w1_parent=d, best_mult=best_mult))
    return out


def report(d):
    named = d[d.rule.isin(list(RULES) + ['parent_optimal'])]
    pd.set_option('display.width', 220)
    print('=' * 78)
    print('W1 AGAINST THE PARENT, by bandwidth rule')
    print('=' * 78)
    for wt in ('Uniform', 'Variable'):
        g = named[named.weighting == wt]
        t = g.groupby('rule').agg(
            median_h_over_sd=('h_over_sd', 'median'),
            mean_w1_parent=('w1_parent', 'mean'),
            median_w1_parent=('w1_parent', 'median'),
            p90_w1_parent=('w1_parent', lambda s: s.quantile(0.90)),
            mean_w1_in_sample=('w1_in_sample', 'mean'),
            mean_loo=('loo', 'mean')).sort_values('mean_w1_parent')
        print(f'--- {wt} weighting, sorted by the PARENT referee (lower is better) ---')
        print(t.to_string(float_format=lambda v: f'{v:.4f}'))
        wide = g.pivot_table(index='dataset', columns='rule', values='w1_parent')
        if set(RULES) <= set(wide.columns):
            print('  head-to-head, share of datasets each rule wins among the three:')
            win = wide[list(RULES)].idxmin(axis=1).value_counts(normalize=True)
            for r in RULES:
                print(f'    {r:<20} {100 * win.get(r, 0.0):5.1f} pct')
            print(f'  guarded beats Scott on          '
                  f'{100 * (wide.silverman_guarded < wide.scott).mean():5.1f} pct')
            print(f'  guarded beats pure Silverman on '
                  f'{100 * (wide.silverman_guarded < wide.silverman).mean():5.1f} pct')
            print(f'  excess over the parent-optimal bandwidth, median ratio:')
            for r in RULES:
                print(f'    {r:<20} '
                      f'{float((wide[r] / wide.parent_optimal).median()):6.3f}x')
        print()
    print('THE ONE QUESTION THIS SCRIPT EXISTS TO ANSWER: where does the PARENT')
    print('criterion put its optimal bandwidth, as a multiple of Scott\'s? In')
    print('sample W1 puts it at 0.02 for most datasets, which is the artifact.')
    b = named.drop_duplicates(['dataset', 'weighting'])[['weighting', 'n',
                                                         'best_mult']]
    print(b.groupby('weighting').best_mult.describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_string(
        float_format=lambda v: f'{v:.4f}'))
    print()
    print('  share of datasets whose parent-optimal multiple is at the sweep floor '
          f'({SWEEP[0]:.2f}): '
          f'{100 * float((b.best_mult <= SWEEP[0] * 1.001).mean()):.1f} pct')
    print('  by size band:')
    bb = b.assign(band=b.n.map(R.size_band))
    print(bb.groupby(['weighting', 'band']).best_mult.median().to_string(
        float_format=lambda v: f'{v:.4f}'))


def main(n_synth):
    os.makedirs(TABLES, exist_ok=True)
    rng = np.random.default_rng(0)
    met, vals, _ = corpus.load_corpus()
    pick = sorted(met.sample(n_synth, random_state=0).dataset.astype(str))
    syn = corpus.as_dict(vals[vals.dataset_id.isin(set(pick))])
    specs = corpus.load_parent_specs()
    print(f'synthetic: {len(syn)} datasets', flush=True)
    rows = []
    for i, name in enumerate(pick):
        x, w = syn[name]
        parent = M.parent_from_spec(specs[name])
        for wt in ('Uniform', 'Variable'):
            rows += one(name, x, w, parent, wt, rng)
        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{len(pick)}', flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(TABLES, 'TABLE_BandwidthAgainstParent.csv'),
             index=False)
    print(f'wrote TABLE_BandwidthAgainstParent.csv ({len(d)} rows)')
    report(d)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_SYNTH)
