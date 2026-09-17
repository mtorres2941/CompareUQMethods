"""How much does the weighting scheme matter, per dataset? Stage 2d.

Four measurements, in the order they have to be made.

1. THE DECOMPOSITION. W1 between the uniform-weighted and the variable-weighted
   version of a dataset is bounded below by the difference in their means. Split
   it, and report how much of the distance the bound carries. If it is most of
   it, the practitioner rule needs no distributional machinery.

2. THE RELATIVE MEASURE. Every dataset is divided by its own unweighted mean, so
   every W1 in this study is already relative to a mean. Name it, verify it
   against an UN-NORMALIZED rerun, and compare it with two robust denominators.

3. A_IQR, from the author's own KL2 paper. The area of the interquartile range
   of the ensemble of densities produced by sampling Dirichlet weights.

4. WHAT DRIVES IT. This is the one place in the study where dispersion beats
   dataset size, which is the opposite of every other finding in the project.

    conda run -n compareuq python audits/weighting_measure.py [n_draws]
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

import corpus  # noqa: E402
import empirical  # noqa: E402
import genconfig as G  # noqa: E402
import recovery as R  # noqa: E402
import weighting as WG  # noqa: E402
from datageneration import clean_empirical_symmetric  # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'audits')
SEED = 20260917

#: Synthetic datasets per size band. A_IQR at 1,000 draws costs about a second
#: per dataset, so the whole corpus would be hours; a stratified sample of this
#: size settles the relationships to three decimal places and is reported with
#: its own size.
SYNTH_PER_BAND = 250
FMT = lambda v: f'{v:.4f}'  # noqa: E731


def head(t):
    print()
    print('=' * 78)
    print(t)
    print('=' * 78)


def decompose(datasets):
    rows = []
    for name, (x, w) in datasets.items():
        d = WG.location_shape_split(x, w)
        d.update(dataset=name, n=len(x),
                 coeffvar=float(np.std(x) / np.mean(x)))
        rows.append(d)
    return R.add_size_band(pd.DataFrame(rows))


def main(n_draws=WG.N_DRAWS):
    os.makedirs(TABLES, exist_ok=True)
    n_draws = int(n_draws)
    rng = np.random.default_rng(SEED)
    alpha = G.DEFAULT.point_weight_alpha
    emp, _ = empirical.prepare(rng.spawn(1)[0])
    metrics, values, meta = corpus.load_corpus()
    SYN = corpus.as_legacy_dict(metrics, values)
    syn = {k: (v['data'], v['weights']) for k, v in SYN.items()}
    print(f'empirical {len(emp)}   synthetic {len(syn):,}   '
          f'corpus {meta["label"]}   alpha {alpha}   draws {n_draws}')

    # ------------------------------------------------------------- 1. split
    de, ds = decompose(emp), decompose(syn)
    de.insert(0, 'arm', 'empirical')
    ds.insert(0, 'arm', 'synthetic')
    split = pd.concat([de, ds], ignore_index=True)
    split.to_csv(os.path.join(TABLES, 'TABLE_WeightingDecomposeW1.csv.gz'),
                 index=False)
    head('1. IS THE UNIFORM-TO-VARIABLE W1 A SHIFT OF THE MEAN, OR A CHANGE OF SHAPE?')
    print('W1 >= |difference in means|. The location term IS that bound; shape')
    print('is the residual. A location share near 1 means reweighting only')
    print('moved the mean.')
    print()
    print(split.groupby('arm')[['w1', 'location', 'shape', 'location_share']]
          .agg(['mean', 'median']).to_string(float_format=FMT))
    print()
    print('  by size band, median:')
    print(split.groupby(['arm', 'size_band'], observed=True)
          [['n', 'w1', 'location', 'shape', 'location_share']].median()
          .to_string(float_format=FMT))
    print()
    for arm, g in split.groupby('arm'):
        print(f'  {arm:10s} location share: median {g.location_share.median():.4f}, '
              f'mean {g.location_share.mean():.4f}, '
              f'pooled sum(location)/sum(w1) {g.location.sum() / g.w1.sum():.4f}, '
              f'share above 0.5: {(g.location_share > 0.5).mean():.4f}')
    print()
    print('  worst violations of the inequality (must be >= 0, floating point '
          'only):')
    print(f'  min residual w1 - location = {(split.w1 - split.location).min():.3e}')

    # ------------------------------- 2. the relative measure, and invariance
    head('2. THE RELATIVE MEASURE, AND WHETHER NORMALIZATION MATTERS')
    print('Every dataset is divided by its unweighted mean before anything')
    print('else, so a reported W1 is already W1 / mean. The check is that')
    print('rerunning on the RAW, un-normalized values reproduces it.')
    raw, _ = empirical.load_raw(empirical.SOURCE, split=empirical.SPLIT,
                               ceiling=True)
    rows = []
    for name in sorted(emp):
        if name not in raw:
            continue
        x_norm, w = emp[name]
        x_raw = np.asarray(clean_empirical_symmetric(raw[name],
                                                     empirical.CLEAN_IQR_MULT),
                           dtype=float)
        if len(x_raw) != len(x_norm):
            continue
        # The weights are keyed by dataset name, so the same vector applies to
        # both scalings; only the x axis differs.
        a = WG.location_shape_split(x_norm, w)
        b = WG.location_shape_split(x_raw, w)
        sa = WG.relative_scales(x_norm)
        sb = WG.relative_scales(x_raw)
        ra = WG.relativize(a['w1'], sa)
        rb = WG.relativize(b['w1'], sb)
        rows.append(dict(dataset=name, n=len(x_raw),
                         data_mean=float(np.mean(x_raw)),
                         w1_normalized=a['w1'], w1_raw=b['w1'],
                         **{f'{k}_norm': ra[k] for k in ra},
                         **{f'{k}_raw': rb[k] for k in rb}))
    inv = pd.DataFrame(rows)
    inv.to_csv(os.path.join(TABLES, 'TABLE_WeightingScaleInvariance.csv'),
               index=False)
    print(f'\n  {len(inv)} datasets rerun on raw values, means spanning '
          f'{inv.data_mean.min():.3e} to {inv.data_mean.max():.3e}')
    print('  worst relative disagreement between the normalized and the raw run:')
    for k in WG.SCALES:
        a, b = inv[f'rel_{k}_norm'], inv[f'rel_{k}_raw']
        rel = np.abs(a - b) / np.abs(a).replace(0, np.nan)
        print(f'    rel_{k:5s} {np.nanmax(rel):.3e}')
    print('  and the ABSOLUTE W1, which is NOT invariant and should not be:')
    print(f'    max ratio w1_raw / w1_normalized = '
          f'{(inv.w1_raw / inv.w1_normalized).max():.3e}')

    # ---- which denominator is more stable across weight draws
    print('\n  WHICH DENOMINATOR BEHAVES MORE STABLY. Dividing one numerator by')
    print('  three constants cannot change how the ratio moves with the weight')
    print('  draw, so the question is not about the numerator at all: it is how')
    print('  precisely each DENOMINATOR can itself be estimated from n values.')
    print('  Measured by resampling the data values, as a coefficient of')
    print('  variation of the denominator. Lower is more stable.')
    stab = stability(emp, np.random.default_rng(99))
    stab.to_csv(os.path.join(TABLES, 'TABLE_WeightingScaleStability.csv'),
                index=False)
    print()
    print(stab.groupby('size_band', observed=True)
          [[f'cv_{k}' for k in WG.SCALES]].median().to_string(float_format=FMT))
    print()
    print('  over all datasets, median bootstrap spread of the denominator,')
    print('  and how often it is undefined or zero:')
    for k in WG.SCALES:
        print(f'    {k:5s} cv {stab[f"cv_{k}"].median():.4f}   '
              f'degenerate on {int(stab[f"zero_{k}"].sum())} datasets')
    print()
    print('  The decisive test is not here but in audits/flip_calibration.py,')
    print('  which asks which denominator makes the flip curve tightest.')

    # ---------------------------------------------------------- 3. A_IQR
    head('3. A_IQR, the KL2 measure')
    print('KL2: "the area of the IQR across all viable PDFs, AIQR, which is')
    print('calculated by subtracting the 25th percentile density curve from the')
    print('75th percentile density curve at each point along the x-axis."')
    print('Read off the paper by Stage 2d: NOT normalized (KL2 reports bare')
    print('areas of 0.40, 0.22 and 0.12), and 1,000 Dirichlet draws.')
    t0 = time.time()
    rows = []
    for name in sorted(emp):
        x = np.asarray(emp[name][0], dtype=float)
        r = WG.dataset_risk(x, np.random.default_rng(abs(hash(name)) % 2 ** 31),
                            n_draws=n_draws, alpha=alpha)
        r.update(arm='empirical', dataset=name,
                 coeffvar=float(np.std(x) / np.mean(x)))
        rows.append(r)
    print(f'  empirical arm in {time.time() - t0:.0f}s')

    t0 = time.time()
    sub = stratified_sample(metrics, np.random.default_rng(5), SYNTH_PER_BAND)
    for name in sub:
        x = np.asarray(syn[name][0], dtype=float)
        r = WG.dataset_risk(x, np.random.default_rng(abs(hash(name)) % 2 ** 31),
                            n_draws=n_draws, alpha=alpha)
        r.update(arm='synthetic', dataset=name,
                 coeffvar=float(np.std(x) / np.mean(x)))
        rows.append(r)
    print(f'  {len(sub):,} synthetic datasets in {time.time() - t0:.0f}s')

    risk = R.add_size_band(pd.DataFrame(rows))
    risk = risk.merge(split[['arm', 'dataset', 'location_share', 'w1']],
                      on=['arm', 'dataset'], how='left')
    risk.to_csv(os.path.join(TABLES, 'TABLE_WeightingAIQR.csv'), index=False)
    print()
    print(risk.groupby(['arm', 'size_band'], observed=True)
          [['n', 'aiqr', 'coeffvar', 'sep_median_mean']].median()
          .to_string(float_format=FMT))

    # ------------------------------------------------- 4. what drives it
    head('4. WHAT DRIVES IT: the one place dispersion beats size')
    print('TWO MEASURES, AND THEY DO NOT AGREE ABOUT WHAT DRIVES THE RISK.')
    print('  aiqr             KL2\'s area of the density interquartile band')
    print('  sep_median_mean  the median distance, over draws, between the')
    print('                   uniform-weighted fit and the drawn one, in units')
    print('                   of the dataset mean. This is the axis the flip')
    print('                   probability is calibrated on.')
    rows = []
    for arm, g in risk.groupby('arm'):
        print(f'\n  {arm} ({len(g)} datasets), Spearman:')
        print(f'    {"":16s} {"A_IQR":>9s} {"sep_median_mean":>17s}')
        for c, lab in (('coeffvar', 'coeffvar'), ('n', 'n'),
                       ('logn', 'log n'), ('w1', 'w1, one draw')):
            col = np.log(g.n) if c == 'logn' else g[c]
            a = pd.DataFrame({'a': g.aiqr, 'b': col}).corr('spearman').iloc[0, 1]
            b = pd.DataFrame({'a': g.sep_median_mean, 'b': col}
                             ).corr('spearman').iloc[0, 1]
            print(f'    {lab:16s} {a:+9.3f} {b:+17.3f}')
            rows.append(dict(arm=arm, against=lab, aiqr=a,
                             sep_median_mean=b))
    pd.DataFrame(rows).to_csv(
        os.path.join(TABLES, 'TABLE_WeightingRiskDrivers.csv'), index=False)
    print()
    print('  WHY THEY DISAGREE, and it is structural rather than a defect.')
    print('  A_IQR is an area under a difference of DENSITIES, and a density')
    print('  carries units of 1/x, so the integral is dimensionless and exactly')
    print('  invariant under rescaling the data. A measure that cannot see a')
    print('  change of scale cannot see dispersion either. The separation in')
    print('  units of the mean is not scale-free in that sense and does see it.')
    print('  So A_IQR answers "how much does the density wobble" and the')
    print('  mean-relative separation answers "how far does the answer move".')
    print('\n  riskiest and safest empirical categories by A_IQR:')
    e = risk[risk.arm == 'empirical']
    cols = ['dataset', 'n', 'coeffvar', 'aiqr']
    print(e.nlargest(6, 'aiqr')[cols].to_string(index=False, float_format=FMT))
    print(e.nsmallest(6, 'aiqr')[cols].to_string(index=False, float_format=FMT))


def stability(datasets, rng, resamples=400):
    """How precisely each candidate denominator can be estimated, per dataset.

    Bootstrap the VALUES of the dataset and recompute the three scales. The
    interquartile range is a consistent but inefficient estimator of spread --
    its asymptotic relative efficiency under normality is about 37 percent --
    and at n = 3 to 9 its quartiles are interpolated between two order
    statistics, which is the same weakness the KDE bandwidth guard exists to
    handle. The mean has no such problem. `zero_*` counts datasets where the
    denominator is zero or non-finite and the relative measure is undefined.
    """
    rows = []
    for name, (x, _) in datasets.items():
        x = np.asarray(x, dtype=float)
        n = len(x)
        base = WG.relative_scales(x)
        boot = [WG.relative_scales(rng.choice(x, size=n, replace=True))
                for _ in range(resamples)]
        row = dict(dataset=name, n=n)
        for k in WG.SCALES:
            v = np.array([b[k] for b in boot], dtype=float)
            m = float(np.mean(v))
            row[f'{k}'] = base[k]
            row[f'cv_{k}'] = float(np.std(v) / m) if m > 0 else np.nan
            row[f'zero_{k}'] = int(not (base[k] > 0) or (v <= 0).any())
        rows.append(row)
    return R.add_size_band(pd.DataFrame(rows))


def stratified_sample(metrics, rng, per_band):
    out = []
    for label, lo, hi in R.SIZE_BANDS:
        pool = metrics[(metrics.n >= lo) & (metrics.n <= hi)].dataset.to_numpy()
        take = min(per_band, len(pool))
        out += list(rng.choice(pool, size=take, replace=False))
    return sorted(out)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else WG.N_DRAWS)
