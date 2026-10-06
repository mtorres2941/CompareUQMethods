"""What the paper is actually competing against: the TWO-parameter lognormal.

WHY THIS EXISTS. The claim scorecard puts the kernel estimate and this study's
three-parameter lognormal within a point or two of each other on most claims,
and the author read that as "the lognormal was right all along and my research
was a side quest back to what people were already doing". That reading turns on
an assumption worth testing: that the lognormal this study fits is the lognormal
the field uses. It is not.

WHAT THE FIELD USES is a TWO-parameter lognormal -- a geometric mean and a
geometric standard deviation. It is ecoinvent's default and it is what the
pedigree matrix produces, since a GSD is a lognormal parameterization.

WHAT THIS STUDY FITS is a THREE-parameter lognormal whose threshold is chosen
by PROFILE LIKELIHOOD, because the global maximum likelihood estimate does not
exist: the likelihood is unbounded as the threshold approaches the smallest
observation. That needed machinery -- `families.fit_lognorm3_profile`, the
guard `PROFILE_DELTA_LO_FRAC = 0.25` calibrated on a bounded-variance criterion
rather than on the score it is judged by (decision 51), and an explicit
truncation to (0, inf) with renormalization (decision 50). The guard determines
the threshold for 48 percent of empirical fits.

So this scores all five families against the KNOWN PARENT, equal weights, and
reports each one's gain over the two-parameter lognormal.

    conda run -n compareuq python audits/lognormal_variants.py [n_datasets]
"""
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import corpus                      # noqa: E402
import empirical as EMP            # noqa: E402
import fitting as FT               # noqa: E402
import recovery as R               # noqa: E402

#: short name -> the family name `fitting.FAMILIES` knows it by.
PARAMETRIC = {'lognorm2': 'lognormal_2p', 'lognorm3': 'lognormal_3p',
              'gamma': 'gamma', 'normal': 'normal'}
BANDS = ([0, 9, 99, 999, 10 ** 9], ['3-9', '10-99', '100-999', '1000+'])
OUT = os.path.join(ROOT, 'outputs', 'tables', 'audits')


def score(n_datasets=1500, seed=7):
    metrics, values, _ = corpus.load_corpus()
    pick = metrics.sample(min(n_datasets, len(metrics)),
                          random_state=seed).dataset.tolist()
    parents = corpus.load_parent_objects(datasets=pick)
    grouped = values[values.dataset_id.isin(pick)].groupby('dataset_id')
    rows, t0 = [], time.time()
    for i, ds in enumerate(pick):
        d = grouped.get_group(ds)
        x = d.value.to_numpy(float)
        w = FT.uniform_weights(x)
        grid = FT.score_grid_open(x, w)
        row = {'dataset': ds, 'n': len(x)}
        for short, family in PARAMETRIC.items():
            model, _params = FT.fit_family(family, x, w)
            row[short] = R.w1_against_parent(model, parents[ds], 'market', grid)
        kde = FT.fit_kde(x, w)
        row['kde'] = R.w1_against_parent(kde[0] if isinstance(kde, tuple)
                                         else kde, parents[ds], 'market', grid)
        rows.append(row)
        if i and i % 500 == 0:
            print(f'  {i} / {len(pick)}  {time.time() - t0:.0f}s', flush=True)
    return pd.DataFrame(rows)


def main(argv):
    n = int(argv[1]) if len(argv) > 1 else 1500
    r = score(n).dropna()
    r['band'] = pd.cut(r.n, BANDS[0], labels=BANDS[1])
    cols = list(PARAMETRIC) + ['kde']
    for c in cols:
        if c != 'lognorm2':
            r[f'{c}_rel'] = 100.0 * (r[c] - r.lognorm2) / r.lognorm2

    print(f'\n{len(r)} corpus datasets, scored against the KNOWN PARENT under '
          'equal weights.\n')
    print('median W1 against the parent:')
    print(r.groupby('band', observed=True)[cols].median()
          .to_string(float_format=lambda v: f'{v:.4f}'))
    print('\nshare of datasets each family is CLOSEST on, pct:')
    print(' ', (100 * r[cols].idxmin(axis=1)
                .value_counts(normalize=True)).round(1).to_dict())
    print('\nmedian relative gain over the TWO-PARAMETER lognormal, by band '
          '(negative = better than it):')
    print(r.groupby('band', observed=True)[[f'{c}_rel' for c in cols
                                            if c != 'lognorm2']].median()
          .to_string(float_format=lambda v: f'{v:+.1f}'))
    print('\nshare of datasets closer to the parent than the two-parameter '
          'lognormal:')
    for c in cols:
        if c == 'lognorm2':
            continue
        print(f'  {c:9s} {100 * (r[c] < r.lognorm2).mean():5.1f} pct')
    # Provenance, per decision 202: which corpus and which weight rule. An
    # earlier version of this table carried neither, and the Stage 2h result
    # taken from it went stale when the corpus was regenerated without anyone
    # being able to tell from the file.
    r['corpus'] = os.path.basename(corpus.active_dir()).replace('corpus_', '')
    r['weight_rho'] = float(EMP.WEIGHT_RHO)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'TABLE_LognormalVariants.csv')
    r.to_csv(path, index=False)
    print(f"\nwrote {path}  (corpus {r['corpus'].iloc[0]}, "
          f"weight_rho {r['weight_rho'].iloc[0]})")


if __name__ == '__main__':
    main(sys.argv)
