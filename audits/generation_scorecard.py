"""How far is the corpus from real data, right now, in under a minute.

THE ITERATION LOOP FOR DATA GENERATION IS NOT THE NOTEBOOK. Notebook 3 takes
about fifty minutes because it runs 2,500 probabilistic LCAs; it has nothing to
do with whether the generated datasets look like real ones. Scoring the
generator against the empirical arm is `tune_configuration.sample_config`, which
is 75 seconds at the full pre-flight scale and about 20 at the scale this
script defaults to. Nothing here writes a corpus and nothing regenerates.

WHAT IT PRINTS. Every characteristic, ranked by how far apart the two arms'
DISTRIBUTIONS sit, with BOTH scalings:

    absolute      the plain Wasserstein distance between the two
                  distributions of that characteristic
    standardized  the same, divided by the empirical standard deviation of
                  that characteristic across categories

**NEITHER IS "THE" ANSWER AND THE PAIR IS THE POINT.** The characteristics are
in incomparable units -- the distance in dataset SIZE runs to hundreds while
one in a Shapiro statistic runs to hundredths -- so only the standardized
column can be compared ACROSS rows or averaged. But the standardized column
moves when the empirical spread moves, and this project has been misled by that
once already (decision 63), so the absolute column is printed beside it and a
row where the two DISAGREE IN SIGN against the last run is flagged.

**"WORST" HERE MEANS FURTHEST APART AND CARRIES NO OTHER JUDGMENT.** It is not
a claim that the characteristic matters least or most.

HISTORY. Every run appends one row per characteristic to
`outputs/tables/audits/TABLE_GenerationScorecardHistory.csv`, tagged with the
configuration and the time, so successive runs can be read as a trend rather
than as isolated numbers. Pass `--label` to name a run.

    conda run -n compareuq python audits/generation_scorecard.py
    conda run -n compareuq python audits/generation_scorecard.py --full
    conda run -n compareuq python audits/generation_scorecard.py \
        --set min_q1_over_iqr=0.05 --label "floor 0.05"
"""
import argparse
import datetime as _dt
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy import stats as _st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, HERE)

import coverage                      # noqa: E402
import genconfig as G                # noqa: E402
import tune_configuration as TC      # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'audits')
HISTORY = os.path.join(TABLES, 'TABLE_GenerationScorecardHistory.csv')
#: The empirical arm is FROZEN (decision 44) and its characteristics are
#: deterministic given the seed, so re-measuring them on every iteration is
#: 27 seconds spent reproducing the same numbers. Cached here and invalidated
#: by deleting the file.
EMP_CACHE = os.path.join(TABLES, '_empirical_arm_cache.parquet')
EMP_CACHE_MODES = os.path.join(TABLES, '_empirical_arm_cache_modes.npz')

#: Datasets per size stratum. 40 is about 20 seconds and is enough to see a
#: characteristic move; 110 is the pre-flight scale the tuning sweep uses and
#: is what a decision should rest on.
FAST_PER_STRATUM = 40
FULL_PER_STRATUM = TC.PER_STRATUM

#: The objective's seed-to-seed standard deviation, from decisions 47, 49, 61,
#: 63 and 138. A move smaller than this is not a move.
OBJECTIVE_SEED_SD = 0.0066


def score(cfg, per_stratum, emp_met, emp_modes, emp_vis, seed=TC.SEED):
    t0 = time.time()
    syn_met, syn_modes, syn_vis = TC.sample_config(cfg, seed=seed,
                                                   per_stratum=per_stratum)
    d = coverage.distribution_comparison(emp_met, syn_met)
    rows = []
    for m in d.metric:
        e, s = coverage._clean(emp_met[m]), coverage._clean(syn_met[m])
        rows.append(dict(metric=m,
                         absolute=float(_st.wasserstein_distance(e, s)),
                         empirical_median=float(e.median()),
                         synthetic_median=float(s.median())))
    out = (pd.DataFrame(rows)
           .merge(d[['metric', 'w1_standardized', 'ks']], on='metric')
           .rename(columns={'w1_standardized': 'standardized'})
           .sort_values('standardized', ascending=False)
           .reset_index(drop=True))
    _, _, summary = TC.score(emp_met, emp_modes, syn_met, syn_modes,
                             emp_vis, syn_vis)
    return out, summary, time.time() - t0, len(syn_met)


def cached_empirical_arm():
    """The empirical arm's characteristics, measured once and reused.

    The extract is frozen (decision 44) and `TC.empirical_arm` is deterministic
    given its seed, so this returns identical numbers and saves 27 seconds of
    every iteration. Delete the cache file to force a re-measure; the columns
    are checked on load so a stale cache from a different characteristic set
    cannot be used silently.
    """
    if os.path.exists(EMP_CACHE) and os.path.exists(EMP_CACHE_MODES):
        met = pd.read_parquet(EMP_CACHE)
        z = np.load(EMP_CACHE_MODES)
        print(f'  empirical arm from cache '
              f'({os.path.relpath(EMP_CACHE, ROOT)}; delete it to re-measure)')
        return met, z['modes'], z['vis']
    print('measuring the empirical arm, once ...', flush=True)
    met, modes, vis = TC.empirical_arm()
    met.to_parquet(EMP_CACHE)
    np.savez(EMP_CACHE_MODES, modes=modes, vis=vis)
    return met, modes, vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', action='store_true',
                    help='the pre-flight scale, 440 datasets, about 75 seconds')
    ap.add_argument('--label', default='', help='a name for this run')
    ap.add_argument('--set', action='append', default=[], metavar='K=V',
                    help='override a generator parameter, repeatable')
    ap.add_argument('--seed', type=int, default=TC.SEED,
                    help='the generator seed. CHANGING A PARAMETER ALREADY '
                         'diverges the random stream, so two configs differ by '
                         'the parameter AND by a different realized sample. '
                         'Repeat one config at several seeds to measure how '
                         'much of a difference is the sample.')
    args = ap.parse_args()

    os.makedirs(TABLES, exist_ok=True)
    pd.set_option('display.width', 200)
    cfg = G.DEFAULT
    changed = {}
    for item in args.set:
        k, v = item.split('=', 1)
        changed[k] = float(v) if '.' in v or 'e' in v.lower() else int(v)
    if changed:
        cfg = cfg.replace(**changed)
    label = args.label or (', '.join(f'{k}={v}' for k, v in changed.items())
                           or 'current default')
    per_stratum = FULL_PER_STRATUM if args.full else FAST_PER_STRATUM

    emp_met, emp_modes, emp_vis = cached_empirical_arm()
    print(f'  {len(emp_met)} real categories')
    print(f'scoring "{label}" at {per_stratum} datasets per stratum ...',
          flush=True)
    out, summary, secs, n_syn = score(cfg, per_stratum, emp_met, emp_modes,
                                      emp_vis, seed=args.seed)

    print()
    print('=' * 78)
    print(f'HOW FAR THE CORPUS SITS FROM REAL DATA -- "{label}"')
    print(f'{n_syn} synthetic datasets against {len(emp_met)} real '
          f'categories, {secs:.0f} seconds')
    print('=' * 78)
    print('Ranked by STANDARDIZED distance, which is the only column that can')
    print('be compared across rows. `absolute` is the undivided distance and is')
    print('there because the standardized one moves when the empirical spread')
    print('moves. "Furthest apart" is all the ranking means.')
    print()
    print(out[['metric', 'standardized', 'absolute', 'empirical_median',
               'synthetic_median', 'ks']]
          .to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print()
    print(f'  objective (mean standardized, plus the mode terms)  '
          f'{summary["weighted_objective"]:.4f}')
    print(f'  seed-to-seed noise on that objective                '
          f'{OBJECTIVE_SEED_SD:.4f}')
    print(f'  visible-mode total variation                        '
          f'{summary["visible_tv"]:.4f}')
    print()
    print(f'FURTHEST APART: {out.iloc[0].metric} at '
          f'{out.iloc[0].standardized:.4f} standardized, '
          f'{out.iloc[0].absolute:.4f} absolute.')

    # -- history, so successive runs read as a trend --------------------
    stamp = _dt.datetime.now().isoformat(timespec='seconds')
    rec = out.assign(run=stamp, label=label, seed=args.seed,
                     per_stratum=per_stratum,
                     n_synthetic=n_syn,
                     objective=summary['weighted_objective'])
    prior = (pd.read_csv(HISTORY) if os.path.exists(HISTORY)
             else pd.DataFrame())
    pd.concat([prior, rec], ignore_index=True).to_csv(HISTORY, index=False)

    if not prior.empty:
        last = prior[prior.run == prior.run.iloc[-1]]
        cmp = out.merge(last[['metric', 'standardized', 'absolute']],
                        on='metric', suffixes=('', '_prev'))
        cmp['d_std'] = cmp.standardized - cmp.standardized_prev
        cmp['d_abs'] = cmp.absolute - cmp.absolute_prev
        cmp['DISAGREE'] = np.sign(cmp.d_std) != np.sign(cmp.d_abs)
        print()
        print(f'AGAINST THE PREVIOUS RUN ({last.label.iloc[0]}, '
              f'{last.run.iloc[0]}). A row flagged DISAGREE moved one way')
        print('scaled and the other way unscaled, which means the empirical')
        print('spread moved and the comparison is not like for like.')
        print()
        print(cmp[['metric', 'standardized', 'd_std', 'absolute', 'd_abs',
                   'DISAGREE']]
              .to_string(index=False, float_format=lambda v: f'{v:+.4f}'))
    print()
    print(f'history: {os.path.relpath(HISTORY, ROOT)}')


if __name__ == '__main__':
    main()
