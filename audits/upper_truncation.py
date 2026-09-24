"""Capping each fitted model at the top: what it costs and what it buys.

Every model in this study is truncated BELOW at zero, because a negative
emission coefficient is inadmissible and that bound needs no argument. An
upper bound has no equally external anchor -- every dataset here is rescaled to
a mean of 1.0, so the physical ceiling this study applies to the raw
declarations is not a fixed multiple -- which is why Stage 2g stated the option
rather than implementing it and handed the sweep here. Decision 152.

WHAT IT WOULD REMOVE. A distance between two cumulative curves charges for the
MASS a model misplaces and not for how far out it sits, so a model can score
well and still dominate any Monte Carlo that samples from it. Measured in
Stage 2g: scored over the grid alone the criterion returns the same number to
the last digit at every contamination distance beyond the grid's top, while a
material's standard deviation moves by a factor of 517 over the same range.

WHAT ALREADY GUARDS IT, so that the sweep is judged against the right baseline
rather than against nothing. The profile guard on the lognormal threshold
bounds the fit, and the analytic tail term in the criterion charges for what
survives. **The tail term stays in force throughout this sweep**, because
without it the criterion cannot see the failure at all and the sweep would be
measuring nothing.

THE COLUMNS THAT MATTER TOGETHER, which is the constraint Stage 2g attached:
W1 and the largest fitted-model standard deviation, at every cap.

    conda run -n compareuq python audits/upper_truncation.py [n_synth]
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
import empirical              # noqa: E402
import families as F          # noqa: E402
import fitting as FT          # noqa: E402
import recovery as R          # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'audits')
SEED = 42

#: The cap, as a multiple of the largest observation a practitioner holds.
#: `inf` is the study as it stands and is the control.
CAPS = (1.0, 1.5, 2.0, 3.0, 5.0, 10.0, np.inf)

#: A fitted model whose standard deviation exceeds this multiple of the data's
#: own is not a distribution the data support. The same loose threshold the
#: profile-guard sweep uses, for the same reason.
SD_BLOWUP_MULTIPLE = 5.0


def model_sd(m, npoints=20_001):
    return float(np.std(m.ppf(np.linspace(1e-9, 1 - 1e-9, npoints))))


def main(n_synth=500):
    os.makedirs(TABLES, exist_ok=True)
    pd.set_option('display.width', 220)
    ds, _ = empirical.prepare(np.random.default_rng(SEED).spawn(1)[0])
    met, vals, _ = corpus.load_corpus()
    met = met[met.dataset.astype(str).str.startswith('dataset')]
    pick = sorted(met.sample(min(n_synth, len(met)), random_state=0)
                  .dataset.astype(str))
    syn = corpus.as_dict(vals[vals.dataset_id.isin(set(pick))])
    parents = corpus.load_parent_objects(datasets=pick)

    rows = []
    for arm, items in (('empirical', list(ds.items())),
                       ('synthetic', list(syn.items()))):
        for name, (x, w) in items:
            x, w = np.asarray(x, float), np.asarray(w, float)
            base, _ = FT.fit_pewt(x, w)
            parent = parents.get(name) if arm == 'synthetic' else None
            grid = (R.recovery_grid(x, w, parent) if parent is not None
                    else None)
            dsd = float(np.std(x))
            for cap in CAPS:
                capped = F.cap_models(base, x, cap)
                for m, mod in capped.items():
                    row = dict(arm=arm, dataset=str(name), n=len(x),
                               cap=('none' if not np.isfinite(cap)
                                    else float(cap)),
                               method=m,
                               w1=FT.score_w1_model(mod, x, w),
                               model_sd_ratio=(model_sd(mod) / dsd
                                               if dsd > 0 else np.nan))
                    if grid is not None:
                        row['w1_market'] = R.w1_against_parent(
                            mod, parent, 'market', grid)
                    rows.append(row)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(TABLES, 'TABLE_UpperTruncationSweep.csv.gz'),
             index=False)
    report(d)
    return d


def report(d):
    print('=' * 78)
    print('WHAT THE CAP COSTS (W1, in sample) AND WHAT IT BUYS (the largest')
    print("fitted model's spread over the data's own). The tail term in the")
    print('criterion is IN FORCE throughout; without it the criterion cannot')
    print('see the failure the cap removes.')
    print('=' * 78)
    for arm in ('empirical', 'synthetic'):
        a = d[d.arm == arm]
        if a.empty:
            continue
        g = a.groupby('cap').agg(
            mean_w1=('w1', 'mean'), median_w1=('w1', 'median'),
            max_sd_ratio=('model_sd_ratio', 'max'),
            p99_sd_ratio=('model_sd_ratio', lambda s: s.quantile(0.99)),
            pct_blowup=('model_sd_ratio',
                        lambda s: 100.0 * (s > SD_BLOWUP_MULTIPLE).mean()))
        if 'w1_market' in a and a.w1_market.notna().any():
            g['mean_w1_market'] = a.groupby('cap').w1_market.mean()
        print(f'--- {arm} ---')
        print(g.to_string(float_format=lambda v: f'{v:.4g}'))
        print()
    print('BY METHOD, the largest fitted spread at each cap, because the')
    print('failure is concentrated: every one of the worst runaway tails in')
    print('this study is an EQUAL-WEIGHTED fit to a small dataset.')
    print()
    print(d.pivot_table(index='cap', columns='method',
                        values='model_sd_ratio', aggfunc='max')
          .to_string(float_format=lambda v: f'{v:.3g}'))
    print()
    print('AND WHAT IT COSTS ON THE TRUTH, which is the only criterion that')
    print('can say whether the cap makes the model BETTER rather than tamer:')
    s = d[(d.arm == 'synthetic') & d.get('w1_market', pd.Series(dtype=float))
          .notna()] if 'w1_market' in d else pd.DataFrame()
    if not s.empty:
        print(s.pivot_table(index='cap', columns='method', values='w1_market',
                            aggfunc='mean')
              .to_string(float_format=lambda v: f'{v:.5f}'))


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
