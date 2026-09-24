"""Does the headline survive a different definition of the empirical arm?

THE PRIMARY ANALYSIS is EPD-level uniform weighting over the categories
resolved into specifiable products by Stage 2a-3's three metadata rules. Two
variants are reported as sensitivities, and the second matters most.

    deduplicated   one record per (manufacturer, product name). 55 percent of
                   records share a (manufacturer, GWP) pair, so the
                   uniform-weighted baseline is already implicitly weighted by
                   how often a manufacturer publishes. Decision 34 keeps the
                   full set as primary, because it is what a practitioner
                   pulling from EC3 actually holds, and states the implicit
                   weighting in the text.

    unsplit        the original EC3 categories, before the splits. **THIS IS
                   THE ONE THAT MATTERS**, because it is the evidence that
                   resolving categories into products did not manufacture the
                   headline result. It is therefore reported against every
                   headline figure rather than against the aggregate alone.

WHAT A HEADLINE FIGURE IS, here. The empirical arm has no known parent, so
every method comparison on it is cross-validated: fit on half the values,
score on the other half, both directions, several splits, all six methods
sharing each split so the comparison is paired. The figures reported are the
ones the paper states:

    the win share of each method
    the size at which the kernel estimate overtakes the lognormal
    the median uniform-to-variable separation, the paper's central quantity
    the characteristic distribution that the generator is calibrated against

    conda run -n compareuq python audits/empirical_population.py [n_splits]
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

import categorysplit                       # noqa: E402
import empirical                           # noqa: E402
import fitting as FT                       # noqa: E402
import recovery as R                       # noqa: E402
import weighting as WG                     # noqa: E402
from customstats import empirical_metadata  # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'audits')
SEED = 20260924

SIZE_BANDS = (('3-9', 3, 9), ('10-99', 10, 99), ('100-999', 100, 999),
              ('1000+', 1000, None))


def band_of(n):
    for label, lo, hi in SIZE_BANDS:
        if n >= lo and (hi is None or n <= hi):
            return label
    return None


def deduplicated_records():
    """One record per (manufacturer, product name), on the raw extract.

    THE KEY IS METADATA AND NEVER A VALUE, which is the same constraint the
    category rules carry (decisions 43, 46 and 60): this study measures the
    dispersion and modality of ECC distributions, so deduplicating on the
    coefficients would be circular. `manufacturer` and `name` are both record
    metadata.
    """
    raw = pd.read_csv(empirical.SOURCE)
    meta = pd.read_csv(empirical.METADATA,
                       usecols=['open_xpd_uuid', 'name'])
    df = raw.merge(meta, on='open_xpd_uuid', how='left')
    key = (df.manufacturer.fillna('~none~').astype(str).str.strip()
           .str.lower() + '||'
           + df.name.fillna('~none~').astype(str).str.strip().str.lower())
    return df.assign(_key=key).drop_duplicates('_key').open_xpd_uuid


def build_arms(rng):
    """The primary arm and the two sensitivities, each as {name: (x, w)}."""
    arms = {}
    arms['primary'] = empirical.prepare(rng.spawn(1)[0])[0]
    arms['unsplit'] = empirical.prepare(rng.spawn(1)[0], split=False)[0]

    keep = set(deduplicated_records())
    original = empirical.load_records

    def patched(path=empirical.SOURCE, metadata=empirical.METADATA):
        df = original(path, metadata)
        return df[df.open_xpd_uuid.isin(keep)]

    empirical.load_records = patched
    try:
        arms['deduplicated'] = empirical.prepare(rng.spawn(1)[0])[0]
    finally:
        empirical.load_records = original
    return arms


def headline_figures(arms, rng, repeats=6):
    """The cross-validated method comparison on each arm.

    The empirical arm has no known parent, so this is the criterion the paper
    uses on it: fit on half the values, score on the other half, both
    directions, several splits, all six methods sharing each split so the
    comparison is paired. It is undefined below ten values, which
    `recovery.CV_MIN_N` enforces and which the per-arm counts below make
    visible rather than silent.
    """
    out = []
    for arm, ds in arms.items():
        got = R.cv_arm(ds, rng.spawn(1)[0], arm, repeats=repeats,
                       loglik=False)
        out.append(got)
        print(f'  cross-validated {arm}: {got.dataset.nunique()} datasets',
              flush=True)
    return pd.concat(out, ignore_index=True)


def crossover(summary, method_a='KDE, Uniform', method_b='Lognormal, Uniform'):
    """The dataset size at which method A overtakes method B, per arm.

    Fitted as a logistic on log(n) for "is A closer", which is the same shape
    the rest of this project inverts a crossing from, and reported with the
    share of datasets A wins in each size band beside it so a reader can see
    the curve the crossing came from.
    """
    piv = summary.pivot_table(index=['arm', 'dataset', 'n'], columns='method',
                              values='w1_cv')
    piv = piv.dropna(subset=[method_a, method_b]).reset_index()
    piv['a_wins'] = (piv[method_a] < piv[method_b]).astype(float)
    rows = []
    for arm, g in piv.groupby('arm'):
        beta = FL.logistic_fit(g.n.to_numpy(float), g.a_wins.to_numpy(float))
        rows.append(dict(arm=arm, n_datasets=len(g),
                         crossover_n=FL.logistic_crossing(beta, 0.5),
                         win_share=float(g.a_wins.mean())))
    return pd.DataFrame(rows), piv


def main(n_splits=6):
    os.makedirs(TABLES, exist_ok=True)
    pd.set_option('display.width', 220)
    rng = np.random.default_rng(SEED)
    arms = build_arms(rng)
    print('ARM SIZES')
    for a, ds in arms.items():
        tot = sum(len(v[0]) for v in ds.values())
        print(f'  {a:<14s} {len(ds):>4d} datasets, {tot:>7d} values')
    print()

    # characteristics, which is what the generator is calibrated against
    rows = []
    for arm, ds in arms.items():
        for name, (x, w) in ds.items():
            m = empirical_metadata(np.asarray(x, float), np.asarray(w, float))
            m.update(arm=arm, dataset=str(name), band=band_of(len(x)))
            rows.append(m)
    chars = pd.DataFrame(rows)
    chars.to_csv(os.path.join(TABLES, 'TABLE_EmpiricalPopulationChars.csv'),
                 index=False)
    print('CHARACTERISTICS, median by arm. The unsplit arm is the one that')
    print('says whether resolving categories into products manufactured the')
    print('result, so it is reported against every figure and not only here.')
    print()
    cols = [c for c in ('n', 'coeffvar', 'skewness', 'kurtosis',
                        'fit_norm_SF', 'fit_lognorm_SF', 'crit_bw_1',
                        'w_v_uw_wasserstein') if c in chars]
    print(chars.groupby('arm')[cols].median()
          .to_string(float_format=lambda v: f'{v:.4f}'))
    print()
    print('THE PAPER\'S CENTRAL QUANTITY, median separation by size band:')
    print(chars.pivot_table(index='arm', columns='band',
                            values='w_v_uw_wasserstein', aggfunc='median')
          .reindex(columns=[b[0] for b in SIZE_BANDS])
          .to_string(float_format=lambda v: f'{v:.4f}'))

    print()
    print('THE METHOD COMPARISON, cross-validated, on each arm.')
    cv = headline_figures(arms, rng, repeats=n_splits)
    cv.to_csv(os.path.join(TABLES, 'TABLE_EmpiricalPopulationCV.csv.gz'),
              index=False)
    summary = (cv.groupby(['arm', 'dataset', 'n', 'method'], as_index=False)
               .w1_cv.mean())
    print()
    print('Median cross-validated W1 by method and arm. A headline that moves')
    print('between the primary and the UNSPLIT arm is a headline the splitting')
    print('produced.')
    print()
    print(summary.pivot_table(index='method', columns='arm', values='w1_cv',
                              aggfunc='median')
          .to_string(float_format=lambda v: f'{v:.4f}'))
    print()
    print('WIN SHARE: how often each method is closest, by arm.')
    best = (summary.loc[summary.groupby(['arm', 'dataset']).w1_cv.idxmin()]
            .groupby(['arm', 'method']).size().rename('wins').reset_index())
    tot = best.groupby('arm').wins.transform('sum')
    best['share'] = best.wins / tot
    print(best.pivot_table(index='method', columns='arm', values='share')
          .to_string(float_format=lambda v: f'{v:.3f}'))
    print()
    cross, _ = crossover(summary)
    cross.to_csv(os.path.join(TABLES, 'TABLE_EmpiricalPopulationCrossover.csv'),
                 index=False)
    print('THE SIZE CROSSOVER: where the kernel estimate overtakes the')
    print('three-parameter lognormal, both under equal weights.')
    print(cross.to_string(index=False, float_format=lambda v: f'{v:.1f}'))
    return arms, chars, cv


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
