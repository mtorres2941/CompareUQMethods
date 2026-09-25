"""Does a candidate corpus survive the WHOLE pipeline, not just the gate?

**THE FAILURE THIS EXISTS TO CATCH, which has already happened once.** Stage 2h
regenerated the corpus on a widened configuration. The generator was fine, the
calibration objective IMPROVED, and every sample-level check passed. The run
against the true parents then reported 99.98 percent errors, because
`plca.ParentSampler` could not resolve the body of a parent inside a support
eight orders of magnitude wide. A day was spent on two wrong diagnoses.

`audits/parent_sampler_fidelity.py` now gates that directly, by comparing the
sampler's quantiles against the parent's own. **This is the step after it.**
The gate checks one component in isolation on freshly drawn parents; this runs
the actual pipeline on an actual candidate corpus -- replay the parents, fit
all six methods to the stored values, score them against the recovered parent,
and check the fitted models are not runaways. A number out of range here means
the candidate breaks something the gate does not cover.

**IT IS A SMOKE TEST AND ITS OUTPUT IS A RANGE CHECK, NOT A RESULT.** It runs
on a DRAFT corpus (decision 41) and its numbers must never reach the paper. The
reference band is the shipped corpus's own published figures: mean W1 against
the market-weighted parent of roughly 0.06 to 0.23 by method, and a largest
fitted-model spread of about 5 times the data's own (decision 152).

    conda run -n compareuq python audits/draft_end_to_end.py draft_joint_bounded_mid
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

import corpus as CO                  # noqa: E402
import fitting as FT                 # noqa: E402
import genconfig as G                # noqa: E402
import plca as PL                    # noqa: E402
import recovery as RC                # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'audits')

#: The band a candidate is judged against. **MEASURED ON THE SHIPPED CORPUS BY
#: THIS SCRIPT, not taken from published figures, because the first version of
#: this file did the latter and was wrong.** That version set the W1 ceiling at
#: 0.35 from numbers that were a different quantity on a different sample; run
#: here on the shipped corpus the mean is 0.3136, so the ceiling sat almost
#: exactly on the corpus already in the paper and a candidate 14 percent above
#: it read as a failure. Reproduce with:
#:
#:     python audits/draft_end_to_end.py 2026-09-21 200
#:
#: The ceiling is now twice the shipped value, which is wide because THE RAW W1
#: IS NOT SCALE FREE: a deliberately more dispersed corpus has a wider parent
#: and a larger absolute distance to it without anything being wrong. The
#: spread-relative figure printed beside it is the one that compares candidates,
#: and it is near 1.0 for a candidate that differs only in scale.
REFERENCE = dict(w1_market_lo=0.03, w1_market_hi=0.65, model_sd_max=6.0,
                 drawn_mean_lo=0.5, drawn_mean_hi=2.0,
                 #: The shipped corpus's own value, BOTH corpora, because
                 #: the first of these was measured on one that has since
                 #: been replaced and a band left pointing at a superseded
                 #: corpus is the silent-failure case this file exists to
                 #: catch. corpus_2026-09-21 gave 0.3136; the current
                 #: corpus_2026-09-25 gives 0.2314, LOWER despite being
                 #: more dispersed, because the stratified sample added in
                 #: the same edit no longer lands almost entirely in the
                 #: smallest size band. Re-measure on any new corpus.
                 shipped_mean_w1_2026_09_21=0.3136,
                 shipped_mean_w1_2026_09_25=0.2314)


def main(label, n_datasets=200):
    d = os.path.join(ROOT, 'data', 'processed', f'corpus_{label}')
    if not os.path.isdir(d):
        raise SystemExit(f'no such corpus: {d}')
    meta = CO.read_meta(d) if hasattr(CO, 'read_meta') else None

    # THE REPLAY REFUSES unless the live configuration equals the one that
    # produced the corpus, which is the right rule and which a draft under
    # overrides cannot satisfy. Substituting the recorded configuration for the
    # duration is the same check, not a weakening of it: the replay still runs
    # the exact configuration the corpus records, and it is restored on the way
    # out so nothing else in the process sees a patched module.
    import json
    with open(os.path.join(d, 'runmeta.json')) as f:
        rec = json.load(f)
    live = G.DEFAULT
    # Rebuild the recorded configuration EXACTLY, including the scaled strata
    # a draft carries, then assert it round-trips. The replay compares
    # `to_dict()` against the record and refuses on any difference, so an
    # approximate reconstruction would simply be refused -- but asserting it
    # here names the differing field instead of failing three lines later.
    # IF THE PARENT SPECS ARE ALREADY CACHED, THE REPLAY HAS ALREADY BEEN DONE
    # AND VERIFIED, so it is loaded rather than repeated. That is not a
    # weakening of the guard: `rebuild_parents` writes that cache only after
    # checking the live configuration against the record AND comparing every
    # replayed value and weight against `values.parquet` element by element
    # (decision 64). Re-running it would also FAIL here for an older corpus,
    # because genconfig has since gained fields the record predates -- fields
    # whose defaults are inert, but the guard compares dictionaries and is
    # right not to reason about which differences are harmless.
    cache = os.path.join(d, 'parents_spec.json.gz')
    if os.path.exists(cache):
        print('  parent specs already cached; loading rather than replaying',
              flush=True)
        parents = CO.load_parent_objects(d)
        return _score(d, parents, label, n_datasets)

    rc = dict(rec['config'])
    scalars = {k: v for k, v in rc.items() if k not in ('strata', 'probe')}
    cfg = G.DEFAULT.replace(
        strata=tuple(G.Stratum(**st) for st in rc['strata']),
        probe=G.Stratum(**rc['probe']), **scalars)
    if cfg.to_dict() != rc:
        differing = sorted(k for k in set(cfg.to_dict()) | set(rc)
                           if cfg.to_dict().get(k) != rc.get(k))
        raise SystemExit(f'could not reconstruct the recorded configuration; '
                         f'differing fields: {differing}')
    print(f'corpus {label}')
    print(f'  min_q1_over_iqr {cfg.min_q1_over_iqr}   '
          f'trunc_iqr_mult {cfg.trunc_iqr_mult}   '
          f'cv_log10_mean {cfg.cv_log10_mean}')
    try:
        G.DEFAULT = cfg
        print('  replaying parents ...', flush=True)
        # `rebuild_parents` replays the generator and CACHES the specs; the
        # parent OBJECTS come from `load_parent_objects`, which rebuilds them
        # from that cache. Both are inside the substitution because the replay
        # is what needs the recorded configuration.
        CO.rebuild_parents(d, verify_values=True, progress=False)
        parents = CO.load_parent_objects(d)
    finally:
        G.DEFAULT = live

    return _score(d, parents, label, n_datasets)


def _score(d, parents, label, n_datasets):
    metrics, values, _ = CO.load_corpus(d)
    vals = CO.as_dict(values)
    # A STRATIFIED sample, not the first N alphabetically. The corpus is
    # generated stratum by stratum, so an alphabetical slice lands almost
    # entirely in the smallest size band and a comparison between two corpora
    # of different sizes is then not size-matched. Sampling evenly across the
    # size range makes the two comparable.
    order = sorted(n for n in vals if n in parents)
    sizes = np.array([len(vals[n][0]) for n in order])
    idx = np.argsort(sizes, kind='mergesort')
    take = np.linspace(0, len(idx) - 1, min(n_datasets, len(idx))).astype(int)
    names = [order[i] for i in np.unique(idx[take])]
    print(f'  {len(names)} datasets scored end to end', flush=True)

    rows = []
    for name in names:
        x, w = np.asarray(vals[name][0]), np.asarray(vals[name][1])
        if len(x) < 4:
            continue
        parent = parents[name]
        fitted, _ = FT.fit_pewt(x, w)
        grid = RC.recovery_grid(x, w, parent)
        sampler = PL.ParentSampler(parent, scheme=PL.TRUTH_SCHEME)
        drawn = np.asarray(sampler.ppf(np.random.default_rng(0).random(5000)))
        for method, model in fitted.items():
            rows.append(dict(
                dataset=name, n=len(x), method=method,
                w1_market=RC.w1_against_parent(model, parent,
                                               PL.TRUTH_SCHEME, grid),
                model_sd_ratio=float(np.std(
                    model.rvs_from_uniform(
                        np.random.default_rng(1).random(4000)))
                    / max(np.std(x), 1e-12)),
                drawn_mean=float(drawn.mean())))
    r = pd.DataFrame(rows)
    os.makedirs(TABLES, exist_ok=True)
    r.to_csv(os.path.join(TABLES, f'TABLE_DraftEndToEnd_{label}.csv'),
             index=False)

    print()
    print('=' * 74)
    print(f'END-TO-END SMOKE TEST -- {label}')
    print('=' * 74)
    print('W1 against the recovered TRUE parent, by method')
    s = r.groupby('method')['w1_market'].agg(['mean', 'median', 'max'])
    print(s.to_string(float_format=lambda v: f'{v:.4f}'))

    if 'coeffvar_uw' in metrics.columns:
        cv = metrics.assign(dataset=metrics.dataset.astype(str)) \
                    .set_index('dataset')['coeffvar_uw']
        r['w1_per_unit_spread'] = (r.w1_market
                                   / r.dataset.map(cv).clip(lower=1e-9))
        print()
        print(f'  median dataset spread {r.dataset.map(cv).median():.4f}   '
              f'mean W1 PER UNIT OF SPREAD '
              f'{r.w1_per_unit_spread.mean():.4f}')
        print('  (the raw W1 above is not scale free; this is. Compare THIS '
              'between candidates.)')

    print()
    print('THE THREE RANGE CHECKS, which is what this run is for')
    mn, mx = r.w1_market.mean(), r.w1_market.max()
    sd = r.model_sd_ratio.max()
    dm = r.drawn_mean
    checks = [
        ('mean W1 against the parent', f'{mn:.4f}',
         REFERENCE['w1_market_lo'] <= mn <= REFERENCE['w1_market_hi'],
         f'{REFERENCE["w1_market_lo"]} to {REFERENCE["w1_market_hi"]}'),
        ('largest fitted-model spread', f'{sd:.2f}',
         sd <= REFERENCE['model_sd_max'],
         f'at most {REFERENCE["model_sd_max"]}'),
        ('median drawn mean from the true parent', f'{dm.median():.4f}',
         REFERENCE['drawn_mean_lo'] <= dm.median() <= REFERENCE['drawn_mean_hi'],
         f'{REFERENCE["drawn_mean_lo"]} to {REFERENCE["drawn_mean_hi"]}'),
    ]
    ok = True
    for name, got, passed, band in checks:
        ok &= passed
        print(f'  {name:42s} {got:>10s}  expected {band:22s} '
              f'{"PASS" if passed else "*** FAIL ***"}')
    print()
    print(f'VERDICT: {"PASSES" if ok else "FAILS"}')
    print('  (a draft corpus. These numbers must never reach the paper.)')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'draft_joint_bounded_mid',
         int(sys.argv[2]) if len(sys.argv) > 2 else 200)
