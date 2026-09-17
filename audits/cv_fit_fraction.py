"""Is the KDE's cross-validated deficit at small n an artifact of HALVING?

The author pressed on this: a KDE is the most flexible method under test, so it
losing to a three-parameter lognormal at n = 10-99 does not pass the sniff test.
It is the fourth time the question has been asked and the previous three each
found a real mechanism, so it is asked here as a measurement rather than argued.

THE SUSPICION IS WELL FOUNDED AND SPECIFIC. A 50/50 cross-validation fits on
n/2, so the band labelled n = 10-99 measures a KDE fitted to 5 to 50 values. A
Gaussian KDE's variance is the data's PLUS h^2, so with a rule-of-thumb
bandwidth it is systematically over-dispersed at small n -- 1.20x at n = 10 --
while a lognormal matches the moments. Cross-validation therefore falls hardest
on the method most sensitive to n, and it does so for a reason that has nothing
to do with how that method would be USED: a practitioner fits to all their data,
not half of it.

WHAT THIS MEASURES. The same paired comparison at fit fractions from 0.5 to 0.9.
If the KDE's deficit shrinks as the fitting half grows, the deficit is partly the
protocol. If it does not, it is the method.

THE SYNTHETIC ARM ANSWERS THE SAME QUESTION WITHOUT ANY SPLITTING, and is
reported beside it: `w1_parent` fits on ALL n and scores against the truth.

    conda run -n compareuq python audits/cv_fit_fraction.py [n_synth]
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
import mixture as M  # noqa: E402
import recovery as R  # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'audits')
SEED = 42
FRACTIONS = (0.5, 0.7, 0.8, 0.9)
REPEATS = 10
N_SYNTH = 1_500


def main(n_synth=N_SYNTH):
    os.makedirs(TABLES, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    emp, _ = empirical.prepare(rng.spawn(1)[0])
    met, vals, _ = corpus.load_corpus()
    ids = sorted(met.sample(n_synth, random_state=0).dataset.astype(str))
    syn = corpus.as_dict(vals[vals.dataset_id.isin(set(ids))])
    specs = corpus.load_parent_specs()

    rows = []
    for arm, ds in (('empirical', emp), ('synthetic', syn)):
        for frac in FRACTIONS:
            r = np.random.default_rng(11)
            cv = R.cv_arm(ds, r, arm, repeats=REPEATS, fit_fraction=frac,
                          loglik=False)
            if not len(cv):
                continue
            s = R.cv_summary(cv)
            s['fit_fraction'] = frac
            rows.append(s)
            print(f'  {arm} at fit fraction {frac}: {time.time()-t0:.0f}s',
                  flush=True)
    cv = pd.concat(rows, ignore_index=True)
    cv = R.add_size_band(cv)
    cv.to_csv(os.path.join(TABLES, 'TABLE_CVFitFraction.csv'), index=False)
    report(cv, syn, ids, specs)


def report(cv, syn, ids, specs):
    pd.set_option('display.width', 220)
    print()
    print('=' * 78)
    print('KDE MINUS LOGNORMAL, paired, by fit fraction. Positive = KDE better.')
    print('=' * 78)
    print('A 0.5 fit fraction is the 50/50 split Stage 2c reported. 0.9 is a')
    print('ten-fold cross-validation, which fits on almost all the data.')
    for arm in ('empirical', 'synthetic'):
        for wt in ('Uniform', 'Variable'):
            print(f'--- {arm}, {wt} ---')
            for frac in FRACTIONS:
                g = cv[(cv.arm == arm) & (cv.fit_fraction == frac)
                       & cv.method.isin([f'KDE, {wt}', f'Lognormal, {wt}'])]
                if not len(g):
                    continue
                allb = R.paired_bootstrap(g, 'w1_cv', f'KDE, {wt}',
                                          rng=np.random.default_rng(0))
                b = R.paired_bootstrap(g, 'w1_cv', f'KDE, {wt}',
                                       by=['size_band'],
                                       rng=np.random.default_rng(0))
                bits = ' | '.join(
                    f'{r.size_band[:4]} {r.mean_difference:+.4f}'
                    f'{"*" if r.distinguishable else " "}'
                    for _, r in b.iterrows())
                print(f'  frac {frac}   ALL {allb.mean_difference.iloc[0]:+.4f}'
                      f'{"*" if allb.distinguishable.iloc[0] else " "}   {bits}')
            print()

    print('=' * 78)
    print('AND WITHOUT ANY SPLITTING AT ALL: the synthetic parent')
    print('=' * 78)
    print('`w1_parent` fits on every value and scores against the truth, so the')
    print('halving cannot be the explanation for anything it says.')
    rows = []
    for name in ids:
        x, w = syn[name]
        x = np.asarray(x, float)
        w = np.asarray(w, float)
        parent = M.parent_from_spec(specs[name])
        import fitting as FT
        models, _ = FT.fit_pewt(x, w)
        grid = R.recovery_grid(x, w, parent)
        for label, m in models.items():
            rows.append(dict(
                arm='synthetic', dataset=name, n=len(x), method=label,
                w1_parent=R.w1_against_parent(
                    m, parent, R.PARENT_SCHEME[R.FT_weighting(label)], grid),
                model_sd_ratio=_sd_ratio(m, x, w)))
    d = R.add_size_band(pd.DataFrame(rows))
    for wt in ('Uniform', 'Variable'):
        g = d[d.method.isin([f'KDE, {wt}', f'Lognormal, {wt}'])]
        b = R.paired_bootstrap(g, 'w1_parent', f'KDE, {wt}', by=['size_band'],
                               rng=np.random.default_rng(0))
        bits = ' | '.join(f'{r.size_band[:4]} {r.mean_difference:+.4f}'
                          f'{"*" if r.distinguishable else " "}'
                          for _, r in b.iterrows())
        print(f'  {wt:<9} {bits}')
    print()
    print('THE MECHANISM, if the deficit at small n survives: a Gaussian KDE is')
    print('the data convolved with a kernel, so its variance is the data\'s PLUS')
    print('h^2 and it is systematically OVER-DISPERSED at small n. Measured here')
    print('as the fitted model\'s standard deviation over the data\'s:')
    print(d.groupby(['size_band', 'method']).model_sd_ratio.mean().unstack()
          .to_string(float_format=lambda v: f'{v:.4f}'))


def _sd_ratio(model, x, w, npoints=20_001):
    import fitting as FT
    sd = FT.weighted_std(x, w)
    if not sd > 0:
        return np.nan
    q = np.linspace(1e-9, 1.0 - 1e-9, npoints)
    return float(np.std(model.ppf(q)) / sd)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_SYNTH)
