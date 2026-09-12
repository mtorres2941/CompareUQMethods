"""Are the individual modes realistically wide, and are ADJACENT modes separated?

The author looked at the dataset-examples figure and objected that many
multimodal datasets show two very tight peaks a long way apart, which is not
what a real material category looks like. Two measurements, because the
objection contains two distinct claims.

  mode_cv        each component's standard deviation divided by the mixture's
                 own mean. The datasets are normalized to mean 1, so this is
                 directly the component's coefficient of variation on the scale
                 the analysis sees. A mode at 0.01 is a spike.

                 MEASURED TWO WAYS, and the difference matters. For the
                 synthetic arm the parent record holds the true component
                 widths; for the empirical arm there is no truth and a
                 BIC-selected Gaussian mixture has to stand in. Comparing the
                 first against the second compares two different estimators, so
                 the BIC route is reported for BOTH arms as the comparable
                 number, with the synthetic truth beside it to show how much the
                 estimator distorts.

                 THE BIC ESTIMATE HAS A FLOOR. gmm_em_1d adds reg = 1e-6 to
                 every variance, so no fitted mode can be narrower than
                 sd = 0.001. The empirical arm sits ON that floor at its 5th and
                 25th percentiles, because real EPD data contains piles of
                 identical values - 55 percent of records share a manufacturer
                 and a GWP - and a Gaussian mixture answers a pile of identical
                 values with a degenerate component. Any empirical mode width at
                 0.0010 is the regularizer, not a measurement, and must not be
                 used as a target.

  min_adj_overlap  the smallest overlap between ADJACENT components, against
                 the average. Adjacent, not all pairs: in one dimension the
                 outermost pair of a five-component mixture is legitimately far
                 apart, so the all-pairs minimum reads 0.0000 on BOTH arms and
                 cannot distinguish them. An earlier version of this script
                 reported the all-pairs minimum and was useless for that reason. This is the
                 known defect recorded in data/INPUTS.sha256 for
                 corpus_2026-09-12: average pairwise overlap barely constrains
                 the modes that actually touch once k > 2, because a few
                 heavily overlapping pairs can carry the average while another
                 pair sits miles apart.

Reported per corpus and against the empirical arm, where the comparable
quantity comes from the same BIC-selected Gaussian mixture used for the overlap
measurement.

    conda run -n compareuq python p9_mode_realism.py [label ...]
"""
import gzip
import json
import os
import sys
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import empirical  # noqa: E402
import mixture as M  # noqa: E402
import modality as MD  # noqa: E402
from scipy import stats  # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'stage2a2')
PROCESSED = os.path.join(ROOT, 'data', 'processed')
SEED = 42
NMAX = 20_000


def pairwise_overlaps(comps, pi):
    """Every pairwise overlap, not just the average the generator targets."""
    out = []
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            w = pi[i] + pi[j]
            if w <= 0:
                continue
            o = M.average_overlap([comps[i], comps[j]],
                                  np.array([pi[i] / w, pi[j] / w]), grid_n=1201)
            out.append(o)
    return out


def synthetic(label):
    parents = json.load(gzip.open(
        os.path.join(PROCESSED, f'corpus_{label}', 'parents.json.gz'), 'rt'))
    met = pd.read_parquet(os.path.join(PROCESSED, f'corpus_{label}',
                                       'metrics.parquet'))
    keep = set(met[~met.is_probe].dataset.astype(str))
    rows = []
    for name, r in parents.items():
        if name not in keep or r.get('status') != 'ok':
            continue
        norm = r['normalizer']
        cvs = [c['sd'] / norm for c in r['components']]
        rows.append(dict(dataset=name, k=r['k'],
                         min_mode_cv=min(cvs), median_mode_cv=float(np.median(cvs)),
                         overlap_achieved=r['overlap_achieved']))
    return pd.DataFrame(rows)


def bic_overlaps(x, rng, kmax=5):
    """Smallest and average pairwise overlap of a BIC-selected mixture.

    Measured from the DATA rather than from the parent record, because the
    record does not store component positions, and because it puts the two arms
    on exactly the same footing: the empirical datasets have no known
    components either.
    """
    x = np.asarray(x, float)
    if len(x) > NMAX:
        x = rng.choice(x, NMAX, replace=False)
    if len(x) < 6 or np.std(x) <= 0:
        return None
    k, pi, mu, sd = MD.fit_mixture_bic(x, rng, kmax=kmax)
    cvs = [s / np.mean(x) for s in sd]
    row = dict(k=int(k), min_mode_cv=float(min(cvs)),
               median_mode_cv=float(np.median(cvs)))
    if k > 1:
        comps = [stats.norm(loc=m, scale=max(s, 1e-9)) for m, s in zip(mu, sd)]
        ovs = pairwise_overlaps(comps, np.asarray(pi, float))
        row.update(avg_overlap=float(np.mean(ovs)),
                   min_overlap=float(np.min(ovs)),
                   min_adj_overlap=float(
                       M.min_adjacent_overlap(comps, np.asarray(pi, float))))
    return row


def synthetic_from_data(label, nsample=500, seed=SEED):
    met = pd.read_parquet(os.path.join(PROCESSED, f'corpus_{label}',
                                       'metrics.parquet'))
    keep = met[~met.is_probe].dataset.astype(str)
    rng = np.random.default_rng(seed)
    pick = set(rng.choice(keep.to_numpy(), min(nsample, len(keep)),
                          replace=False))
    vals = pd.read_parquet(os.path.join(PROCESSED, f'corpus_{label}',
                                        'values.parquet'))
    rows = []
    for dsid, g in vals.groupby('dataset_id', observed=True):
        if str(dsid) not in pick:
            continue
        r = bic_overlaps(g['value'].to_numpy(), rng)
        if r:
            rows.append(r)
    return pd.DataFrame(rows)


def empirical_arm():
    ds, _ = empirical.prepare(np.random.default_rng(SEED).spawn(1)[0])
    rng = np.random.default_rng(SEED)
    rows = []
    for mat, (x, _) in ds.items():
        r = bic_overlaps(x, rng)
        if r:
            rows.append(dict(material=mat, **r))
    return pd.DataFrame(rows)


def q(s, name):
    s = pd.Series(s).replace([np.inf, -np.inf], np.nan).dropna()
    return (f'  {name:<26} p05 {s.quantile(.05):8.4f}   p25 {s.quantile(.25):8.4f}'
            f'   median {s.median():8.4f}   p75 {s.quantile(.75):8.4f}')


if __name__ == '__main__':
    labels = sys.argv[1:] or ['2026-09-12b', '2026-09-12c']

    print('=== empirical, from the BIC-selected mixture ===')
    e = empirical_arm()
    print(q(e.min_mode_cv, 'narrowest mode CV'))
    print(q(e.median_mode_cv, 'median mode CV'))
    if 'min_overlap' in e:
        m = e[e.k > 1]
        print(q(m.min_adj_overlap, 'smallest ADJACENT overlap'))
        print(q(m.avg_overlap, 'average pairwise overlap'))
        print(q(m.min_overlap, '  (all-pairs min, uninformative)'))
    for thr in (0.05, 0.01):
        print(f'  share with a mode CV below {thr}: '
              f'{100*(e.min_mode_cv < thr).mean():5.1f}%')
    print(f'  share AT the reg floor (0.0010): '
          f'{100*(e.min_mode_cv <= 0.00101).mean():5.1f}%   '
          f'<- estimator, not data')
    e.to_csv(os.path.join(TABLES, 'TABLE_2a2_ModeRealism_empirical.csv'),
             index=False)

    for lab in labels:
        print(f'\n=== corpus {lab} ===')
        s = synthetic(lab)
        print('  -- parent record, the TRUE component widths --')
        print(q(s.min_mode_cv, 'narrowest mode CV'))
        print(q(s.median_mode_cv, 'median mode CV'))
        o = synthetic_from_data(lab)
        print('  -- BIC fit, comparable with the empirical arm --')
        print(q(o.min_mode_cv, 'narrowest mode CV'))
        print(q(o.median_mode_cv, 'median mode CV'))
        for thr in (0.05, 0.01):
            print(f'  share with a mode CV below {thr}: '
                  f'{100*(o.min_mode_cv < thr).mean():5.1f}%')
        print(f'  share AT the reg floor (0.0010): '
              f'{100*(o.min_mode_cv <= 0.00101).mean():5.1f}%')
        om = o[o.k > 1] if 'min_overlap' in o else o.iloc[0:0]
        if len(om):
            o = om
            print(q(o.min_adj_overlap, 'smallest ADJACENT overlap'))
            print(q(o.avg_overlap, 'average pairwise overlap'))
            print('  by k, median smallest ADJACENT vs median average:')
            for kk, g in o.groupby('k'):
                print(f'    k={kk}  n={len(g):>4}   '
                      f'min_adj {g.min_adj_overlap.median():.5f}'
                      f'   avg {g.avg_overlap.median():.5f}')
        s.to_csv(os.path.join(TABLES, f'TABLE_2a2_ModeRealism_{lab}.csv'),
                 index=False)
