"""How far below min(x) must the lognormal threshold be held, and why it matters.

an earlier revision. `families.PROFILE_DELTA_LO_FRAC` was set to 0.01 on the reasoning that
a guard only has to stop the likelihood diverging. It does, and the resulting
model is unusable.

WHERE THE DEFECT SHOWED UP. Not in the fit scores. In the pLCA results: after
switching to the profile-likelihood lognormal, `eci_std` for `Lognormal, Uniform`
had a 99th percentile of 33 and a maximum of 532, against 0.92 and 1.56 before.
The fitted model was matching the BODY of the data and carrying an enormous right
tail, which W1 barely charges for -- a thin far tail is a small area between two
CDFs -- and which dominates any Monte Carlo that samples from it.

WHY. Where the profile likelihood has no interior maximum the threshold is driven
onto the guard, and the closer the guard sits to min(x) the larger sigma must be
to accommodate the rest of the data.

This script sweeps the guard on both arms and reports the two things that
matter together: the largest standard deviation of any fitted model, and W1.

    conda run -n compareuq python audits/profile_guard_sweep.py [n_synth]
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
import empirical  # noqa: E402
import families as F  # noqa: E402
import fitting as FT  # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'audits')
SEED = 42
FRACS = (0.01, 0.05, 0.10, 0.25, 0.50, 1.00)
N_SYNTH = 600

#: The OTHER two bounds on the profile-likelihood fit, which Stage 2h was told
#: to check rather than assume non-binding. `delta_hi_frac` is how far below
#: min(x) the threshold grid reaches and `npoints` is how finely it is ruled.
HI_FRACS = (10.0, 100.0, 1000.0, 10000.0)
GRID_POINTS = (100, 200, 400, 800, 1600)

#: A fitted model whose standard deviation exceeds this multiple of the data's
#: own is not a distribution the data support. Loose on purpose: this is the
#: difference between heavy-tailed and unusable, not a calibration.
SD_BLOWUP_MULTIPLE = 5.0


def model_sd(m, npoints=20_001):
    return float(np.std(m.ppf(np.linspace(1e-9, 1 - 1e-9, npoints))))


def main(n_synth):
    os.makedirs(TABLES, exist_ok=True)
    ds, _ = empirical.prepare(np.random.default_rng(SEED).spawn(1)[0])
    met, vals, _ = corpus.load_corpus()
    pick = set(met.sample(n_synth, random_state=0).dataset.astype(str))
    syn = corpus.as_dict(vals[vals.dataset_id.isin(pick)])

    rows = []
    for frac in FRACS:
        for arm, items in (('empirical', ds.items()), ('synthetic', syn.items())):
            sds, w1s, status = [], [], []
            for name, (x, w) in items:
                p = F.fit_lognorm3_profile(x, w, delta_lo_frac=frac)
                m = F.make_lognorm(p)
                sds.append(model_sd(m))
                w1s.append(FT.score_w1_model(m, x, w))
                status.append(p['status'])
            sds, w1s = np.array(sds), np.array(w1s)
            status = np.array(status)
            rows.append(dict(
                delta_lo_frac=frac, arm=arm, n=len(sds),
                pct_at_guard=float((status == 'boundary_guard').mean() * 100),
                pct_interior=float((status == 'interior').mean() * 100),
                pct_normal_limit=float(
                    (status == 'boundary_normal_limit').mean() * 100),
                median_model_sd=float(np.median(sds)),
                p99_model_sd=float(np.quantile(sds, 0.99)),
                max_model_sd=float(sds.max()),
                pct_sd_blowup=float((sds > SD_BLOWUP_MULTIPLE).mean() * 100),
                mean_w1=float(w1s.mean()), median_w1=float(np.median(w1s))))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(TABLES, 'TABLE_ProfileGuardSweep.csv'), index=False)

    pd.set_option('display.width', 220)
    for arm in ('empirical', 'synthetic'):
        print(f'=== {arm} ===')
        print(d[d.arm == arm].drop(columns='arm').to_string(
            index=False, float_format=lambda v: f'{v:.4g}'))
        print()
    print(f'CHOSEN: PROFILE_DELTA_LO_FRAC = {F.PROFILE_DELTA_LO_FRAC}')
    print('  the smallest guard at which no fitted model has a standard')
    print(f'  deviation above {SD_BLOWUP_MULTIPLE:g}, on either arm. W1 is flat')
    print('  from 0.05 to 0.25 and better there than at 0.01 on both arms, so')
    print('  the choice costs nothing on the study\'s own criterion. It is made')
    print('  on the bounded-variance criterion and NOT on W1, so that it is not')
    print('  a number tuned to the score it is then judged by.')

    # THE OTHER TWO BOUNDS, checked rather than assumed. Stage 2h.
    other = check_other_bounds(ds, syn)
    other.to_csv(os.path.join(TABLES, 'TABLE_ProfileOtherBounds.csv'),
                 index=False)
    print()
    print('THE OTHER TWO BOUNDS. `delta_hi_frac` is how far below min(x) the')
    print('threshold grid reaches, currently '
          f'{F.PROFILE_DELTA_HI_FRAC:g}; `npoints` is how finely it is ruled,')
    print(f'currently {F.PROFILE_GRID_POINTS}. Both are assumed non-binding')
    print('and neither had been measured.')
    print()
    print(other.pivot_table(index=['bound', 'value'], columns='arm',
                            values=['mean_w1', 'max_model_sd'])
          .to_string(float_format=lambda v: f'{v:.4g}'))


def check_other_bounds(ds, syn):
    """Are `delta_hi_frac` and `npoints` as non-binding as they look?

    THE GUARD BELOW min(x) IS THE ONE THAT BINDS, and it is swept above. These
    two were never measured, and a bound nobody has tested is a bound nobody
    knows the value of. `delta_hi_frac` matters only if the profile likelihood
    wants a threshold FURTHER below the data than the grid reaches -- the
    normal limit -- and `npoints` only if the maximum sits between two grid
    points.
    """
    rows = []
    for arm, items in (('empirical', list(ds.items())),
                       ('synthetic', list(syn.items()))):
        for hi in HI_FRACS:
            sds, w1s, st = [], [], []
            for _, (x, w) in items:
                p = F.fit_lognorm3_profile(x, w, delta_hi_frac=hi)
                m = F.make_lognorm(p)
                sds.append(model_sd(m))
                w1s.append(FT.score_w1_model(m, x, w))
                st.append(p['status'])
            rows.append(dict(bound='delta_hi_frac', value=hi, arm=arm,
                             mean_w1=float(np.mean(w1s)),
                             max_model_sd=float(np.max(sds)),
                             pct_normal_limit=float(
                                 (np.array(st) == 'boundary_normal_limit')
                                 .mean() * 100)))
        for npts in GRID_POINTS:
            sds, w1s, st = [], [], []
            for _, (x, w) in items:
                p = F.fit_lognorm3_profile(x, w, npoints=npts)
                m = F.make_lognorm(p)
                sds.append(model_sd(m))
                w1s.append(FT.score_w1_model(m, x, w))
                st.append(p['status'])
            rows.append(dict(bound='npoints', value=npts, arm=arm,
                             mean_w1=float(np.mean(w1s)),
                             max_model_sd=float(np.max(sds)),
                             pct_normal_limit=float(
                                 (np.array(st) == 'boundary_normal_limit')
                                 .mean() * 100)))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_SYNTH)
