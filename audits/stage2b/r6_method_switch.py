"""Exactly what the Stage 2b fitting change does to the six reported W1 scores.

Stage 2b. The commit that switches `fit_pewt_models` for `fit_pewt` moves every
W1 in the paper, so the movement is measured here first and recorded in that
commit message. CLAUDE.md: never silently change a result.

Three things change at once and this separates them:

  A. THE GRID is open at zero instead of closed. Discrepancy entry 18.
  B. THE NORMAL AND THE KDE become explicit truncations to (0, inf),
     renormalized, rather than untruncated objects that the grid happened to
     truncate by starting at zero. Decision 13.
  C. THE LOGNORMAL becomes a three-parameter fit whose threshold is chosen by
     profile likelihood, replacing a threshold fixed by hand at -0.5.
     Entries 37 and 40.

A and B are numerically small and C is not. Reporting them together would leave
the author unable to tell which decision moved which number, so each column is
computed against the same data with only that change applied.

    conda run -n compareuq python audits/stage2b/r6_method_switch.py [n_synth]
"""
import os
import sys
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import corpus  # noqa: E402
import empirical  # noqa: E402
import fitting as FT  # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'stage2b')
SEED = 42
N_SYNTH = 2000


def rows_for(name, x, w, arm):
    out = []
    # (A) Stage 1 exactly: untruncated scipy objects, grid closed at zero.
    old = FT.fit_pewt_models(x, w)
    s_old = FT.score_all(old, x, w)
    # (A -> B) the same Stage 1 models, scored on the grid that is open at zero.
    g_open = FT.score_grid_open(x, w)
    s_open = {k: FT.wasserstein1_weighted(x, g_open, w, m.pdf(g_open))
              for k, m in old.items()}
    # (C) the Stage 1 lognormal kept, everything else Stage 2b.
    mid, _ = FT.fit_pewt(x, w, lognormal_family='lognormal_offset')
    s_mid = FT.score_all_models(mid, x, w)
    # Stage 2b as shipped.
    new, params = FT.fit_pewt(x, w)
    s_new = FT.score_all_models(new, x, w)
    for label in FT.PEWT:
        out.append(dict(arm=arm, dataset=name, n=len(x), pewt=label,
                        w1_stage1=s_old[label],
                        w1_open_grid=s_open[label],
                        w1_truncated=s_mid[label],
                        w1_stage2b=s_new[label],
                        lognormal_status=params[label].get('status', '')))
    return out


def main(n_synth):
    os.makedirs(TABLES, exist_ok=True)
    rows = []
    ds, _ = empirical.prepare(np.random.default_rng(SEED).spawn(1)[0])
    print(f'empirical arm: {len(ds)} datasets', flush=True)
    for i, (name, (x, w)) in enumerate(ds.items()):
        rows += rows_for(name, x, w, 'empirical')
        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{len(ds)}', flush=True)

    met, vals, _ = corpus.load_corpus()
    pick = set(met.sample(n_synth, random_state=0).dataset.astype(str))
    syn = corpus.as_dict(vals[vals.dataset_id.isin(pick)])
    print(f'synthetic: {len(syn)} datasets', flush=True)
    for i, (name, (x, w)) in enumerate(syn.items()):
        rows += rows_for(name, x, w, 'synthetic')
        if (i + 1) % 500 == 0:
            print(f'  {i+1}/{len(syn)}', flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(TABLES, 'TABLE_2b_MethodSwitch.csv'), index=False)

    pd.set_option('display.width', 220)
    steps = ['w1_stage1', 'w1_open_grid', 'w1_truncated', 'w1_stage2b']
    labels = {'w1_stage1': 'Stage 1',
              'w1_open_grid': '+ grid open at zero',
              'w1_truncated': '+ explicit truncation',
              'w1_stage2b': '+ profile-likelihood lognormal'}
    for arm in ('empirical', 'synthetic'):
        g = d[d.arm == arm]
        print()
        print('=' * 78)
        print(f'{arm.upper()} ARM, {g.dataset.nunique()} datasets')
        print('=' * 78)
        for agg in ('mean', 'median'):
            print(f'{agg.upper()} W1, one change at a time')
            t = g.pivot_table(index='pewt', values=steps, aggfunc=agg)[steps]
            t.columns = [labels[c] for c in steps]
            print(t.to_string(float_format=lambda v: f'{v:.5f}'))
            print()
        print('MEAN RANK over the six methods')
        for col in steps:
            r = g.pivot(index='dataset', columns='pewt',
                        values=col).rank(axis=1).mean().sort_values()
            print(f'  {labels[col]:<32} ' +
                  '  '.join(f'{k}={v:.2f}' for k, v in r.items()))
        print()
        print('PER-METHOD MOVEMENT, Stage 1 to Stage 2b')
        m = (g.assign(rel=(g.w1_stage2b - g.w1_stage1) / g.w1_stage1 * 100)
             .groupby('pewt').rel.agg(['median', 'mean',
                                       lambda s: (s < 0).mean() * 100]))
        m.columns = ['median_pct', 'mean_pct', 'improved_pct']
        print(m.to_string(float_format=lambda v: f'{v:+.2f}'))

    print()
    print('LOGNORMAL THRESHOLD OUTCOME, both arms')
    print(d[d.pewt.str.startswith('Lognormal')]
          .groupby(['arm', 'lognormal_status']).dataset.count().to_string())


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_SYNTH)
