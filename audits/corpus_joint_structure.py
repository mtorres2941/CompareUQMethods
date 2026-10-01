"""Candidate generator configurations, judged on the JOINT structure.

WHY THIS EXISTS. `audits/corpus_modality_shape.py` established that the corpus
gets the joint structure of modality and shape BACKWARDS: in the real arm more
visible modes come with more spread, skew and kurtosis, and in the corpus they
come with less, on all six characteristics. The calibration objective matches
MARGINAL distributions one characteristic at a time (decision 35) and never
looks at a correlation, so a configuration can match every margin and still be
wrong this way -- which is what happened.

Stage 2f's `dispersion_reach` sweep (decision 138) already tried widening the
dispersion ceiling and judged the candidates on the MARGINAL objective alone.
This re-runs a short version of that sweep and adds the measurement that was
missing: does the candidate also fix the SIGN of the modality-shape
correlations, and at what cost to the margins that currently match?

DRAFT CORPORA ONLY. Each candidate generates a small corpus under a label that
says `draft` (decision 41), and a draft must never supply a paper number.

    conda run -n compareuq python audits/corpus_joint_structure.py [n_datasets]
"""
import dataclasses
import json
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import corpus                       # noqa: E402
import coverage                     # noqa: E402
import genconfig as G               # noqa: E402
import modality as MO               # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables')
OUT = os.path.join(TABLES, 'audits')
CHARS = ['coeffvar', 'skewness', 'kurtosis', 'fit_norm_SF', 'crit_bw_1']

#: The candidates. `trunc_iqr_mult` with `min_q1_over_iqr` is the dispersion
#: ceiling decision 138 identified as the binding constraint -- the achieved
#: coefficient of variation tops out at 1.447 against a real arm reaching 6.93,
#: and 57 percent of drawn targets come back clipped. `overlap_log10_hi` is the
#: modality lever: lower means more separated components and more visible modes.
CANDIDATES = {
    'current': {},
    'wider_dispersion': dict(trunc_iqr_mult=5.0, min_q1_over_iqr=0.02),
    'wider_and_blended': dict(trunc_iqr_mult=5.0, min_q1_over_iqr=0.02,
                              overlap_log10_hi=0.30),
    'widest': dict(trunc_iqr_mult=8.0, min_q1_over_iqr=0.01),
    # THE SECOND ROUND. The first four all widened the DISPERSION ceiling and
    # all four left the sign of the modality-shape correlation negative, so the
    # ceiling is not what flips it. What makes a synthetic multimodal dataset
    # tidy is that its modes are the SAME SIZE: `mode_share_alpha = 10` puts
    # the larger of two modes between 0.50 and 0.76 of the points (decision
    # 27), which is a symmetric two-humped shape, and symmetric shapes have low
    # skew by construction. Real multimodality is a small shoulder on a big
    # skewed body. Dropping the concentration to 1 makes the mode sizes a flat
    # Dirichlet -- the larger mode spans 0.52 to 0.97 -- which is the author's
    # own proposal already carried to Stage 2h in decision 141.
    'unequal_modes': dict(mode_share_alpha=1.0),
    'unequal_and_wider': dict(mode_share_alpha=1.0, trunc_iqr_mult=5.0,
                              min_q1_over_iqr=0.02),
    'unequal_wider_separated': dict(mode_share_alpha=1.0, trunc_iqr_mult=5.0,
                                    min_q1_over_iqr=0.02,
                                    overlap_log10_hi=-0.10),
    # THE THIRD ROUND, and the first that changes the CONSTRUCTION rather than
    # a range. `shoulder_frac` pairs the largest weight and the widest
    # component with the lowest position, so a multi-component parent becomes
    # one dominant body with small narrow components on its upper tail instead
    # of a symmetric blend of separated humps. See genconfig.shoulder_frac.
    'shoulder_half': dict(shoulder_frac=0.5, mode_share_alpha=1.0),
    'shoulder_all': dict(shoulder_frac=1.0, mode_share_alpha=1.0),
    'shoulder_all_wider': dict(shoulder_frac=1.0, mode_share_alpha=1.0,
                               trunc_iqr_mult=5.0, min_q1_over_iqr=0.02),
    'shoulder_all_widest': dict(shoulder_frac=1.0, mode_share_alpha=1.0,
                                trunc_iqr_mult=8.0, min_q1_over_iqr=0.01),
    # THE ARCHITECTURAL PATH, confirmed on the real stratified design. The fast
    # probe samples n log-uniformly and the corpus samples it by stratum, so
    # probe numbers rank candidates and these numbers judge them.
    'separation': dict(separation_dispersion_frac=1.0, mode_share_alpha=1.0,
                       trunc_iqr_mult=5.0, min_q1_over_iqr=0.02),
    'separation_kmin2': dict(separation_dispersion_frac=1.0, k_min=2,
                             mode_share_alpha=1.0, trunc_iqr_mult=5.0,
                             min_q1_over_iqr=0.02),
    # THE FOURTH ROUND, 2026-09-25, and the first judged against an empirical
    # arm weighted by the SAME rule as the corpus (decision 190).
    #
    # TWO THINGS CHANGED UNDER THE FOOT OF EVERY CANDIDATE ABOVE. First, the
    # modality-shape correlations the rounds above were chasing were measured
    # on WEIGHTED characteristics against an empirical arm drawing flat
    # Dirichlet weights; on the weight-invariant UNWEIGHTED columns the real
    # arm shows essentially no modality-shape relationship at all (-0.04 to
    # +0.05 across five characteristics on 130 categories), so "the corpus has
    # the sign backwards" was largely a statement about the old weight draw.
    # What survives is that the corpus has a spurious NEGATIVE relationship
    # (-0.11 to -0.17) where the real data has none. Second, and this is what
    # these candidates are for: conditional on being dispersed, the real arm is
    # multimodal 21.2 percent of the time and the corpus 22.1 -- they AGREE --
    # so the joint cell is short only because the DISPERSION MARGINAL is short
    # by a factor of nine.
    #
    # EVERY CANDIDATE ABOVE THAT WIDENS DISPERSION DOES IT WITH
    # `min_q1_over_iqr = 0.02, trunc_iqr_mult = 5.0`, whose truncation bound
    # multiplier is 345 million and which FAILS the parent-level gate of
    # decision 192: the truth run's sampler is more than 1 percent wrong on 55
    # percent of the parents it makes. Those candidates cannot supply a corpus
    # whatever they score here. These use bounded truncation -- (1 + 1/0.1)**2
    # = 121 and (1 + 1/0.2)**3 = 216, against the shipped 27 -- and pass that
    # gate at 5e-4.
    'bounded_wide': dict(min_q1_over_iqr=0.1, trunc_iqr_mult=2.0,
                         cv_log10_mean=0.329),
    'bounded_mid': dict(min_q1_over_iqr=0.2, trunc_iqr_mult=3.0,
                        cv_log10_mean=0.329),
    'bounded_wide_unequal': dict(min_q1_over_iqr=0.1, trunc_iqr_mult=2.0,
                                 cv_log10_mean=0.329, mode_share_alpha=1.0),
    'bounded_wide_separation': dict(min_q1_over_iqr=0.1, trunc_iqr_mult=2.0,
                                    cv_log10_mean=0.329, mode_share_alpha=1.0,
                                    separation_dispersion_frac=1.0),
    # THE MODALITY THE BOUNDED CANDIDATES GIVE UP. Widening the components
    # blends the humps, so dispersion is bought partly out of visible modality:
    # 21.5 percent multimodal at the shipped configuration against 16.5 to 20.0
    # for the bounded candidates, and a real arm at 24.6. `overlap_log10_hi` is
    # the modality lever -- lower means more separated components -- so these
    # ask whether the loss is recoverable without reaching for the separation
    # path, whose cost to the weighting margin survives the repaired
    # comparison (0.686 against 0.340 at the shipped configuration).
    'bounded_mid_sep_lo': dict(min_q1_over_iqr=0.2, trunc_iqr_mult=3.0,
                               cv_log10_mean=0.329, mode_share_alpha=1.0,
                               overlap_log10_hi=0.10),
    'bounded_mid_sep_lower': dict(min_q1_over_iqr=0.2, trunc_iqr_mult=3.0,
                                  cv_log10_mean=0.329, mode_share_alpha=1.0,
                                  overlap_log10_hi=-0.10),
    # THE FIFTH ROUND, 2026-09-25, and the question is narrow: does the hump
    # SHARE concentration fix the CONDITIONAL modality on the SHIPPED
    # configuration? The corpus now matches the real arm's dispersion far
    # better than it did, and among DISPERSED datasets it is multimodal 14.1
    # percent of the time against the real arm's 21.2 -- so the intersection
    # is where it is worst. `mode_share_alpha` at 1 makes the larger of two
    # modes span 0.52 to 0.97 of the points instead of 0.50 to 0.76, which is
    # a small shoulder on a big body rather than a symmetric pair, and that is
    # what real multimodality looks like (decision 37).
    #
    # It was rejected twice, both times on evidence that no longer stands: at
    # 5.0 GENERATOR-seed standard deviations worse under the mismatched weight
    # rules. Re-run under the settled rule it is +0.0149 against a WEIGHT-DRAW
    # noise of 0.006 to 0.015, which is the boundary, so it is now free on the
    # objective. What was never measured is whether it BUYS anything here.
    'shipped_plus_alpha1': dict(mode_share_alpha=1.0),
    'shipped_plus_alpha1_sep': dict(mode_share_alpha=1.0,
                                    overlap_log10_hi=-0.10),
    # THE SIXTH ROUND, Stage 3, and it closes the one measurement the Stage 3
    # prompt asks for by name. `separation_dispersion_frac` and `shoulder_frac`
    # are the two levers built FOR the joint structure -- the first solves the
    # component SPACING so the mixture itself carries the drawn coefficient of
    # variation, the second pairs the heaviest weight with the widest component
    # so a mode is a shoulder on a body rather than half of a symmetric pair --
    # and both are still at 0.0 in the live configuration and inside
    # corpus_2026-09-25. Every earlier measurement of either was taken under
    # the mismatched weight rules, or on the superseded corpus, or bundled with
    # `mode_share_alpha` so that neither lever could be read on its own.
    #
    # Each is therefore measured ALONE against the shipped configuration, which
    # is what makes the result attributable. Decision 203 settled
    # `mode_share_alpha` and is not reopened here.
    'shipped_plus_separation': dict(separation_dispersion_frac=1.0),
    'shipped_plus_shoulder': dict(shoulder_frac=1.0),
}


def _config_matches(directory, cfg):
    """Was this draft generated under `cfg`?

    Compares every scalar generation parameter recorded in the corpus's own
    runmeta against the configuration about to be asked for. `strata`, `probe`
    and `seed` are excluded: a draft is deliberately scaled down, so those
    differ by construction and say nothing about the shape of the parents.
    """
    import dataclasses
    try:
        rec = json.load(open(os.path.join(directory, 'runmeta.json')))['config']
    except Exception:
        return False
    want = dataclasses.asdict(cfg)
    skip = ('strata', 'probe', 'seed')
    return all(rec.get(k) == v for k, v in want.items()
               if k not in skip and not isinstance(v, (list, tuple, dict)))


def empirical_targets():
    """The signs the corpus has to reproduce, measured on the real arm."""
    vm = pd.read_csv(os.path.join(TABLES, 'TABLE_VisibleModes.csv'))
    e = (pd.read_excel(os.path.join(TABLES,
                                    'TABLE_EmpiricalECCMetricsAndW1.xlsx'))
         .rename(columns={'Unnamed: 0': 'dataset'}))
    e = e.merge(vm[vm.arm == 'empirical'][['dataset', 'modes_fitted']],
                on='dataset')
    return e, {c: spearmanr(e.modes_fitted, e[c])[0] for c in CHARS
               if c in e.columns}


def visible_modes(values, metrics):
    """The mode count at the bandwidth the study fits, per dataset."""
    out = {}
    for ds, g in values.groupby('dataset_id'):
        x = g.value.to_numpy(float)
        if len(x) < 8 or np.std(x) == 0:
            continue
        try:
            out[ds] = MO.n_modes_fitted(x)
        except Exception:
            continue
    return pd.Series(out, name='modes_fitted')


#: "Dispersed" on the weight-invariant column: the real arm's own upper
#: quartile of `coeffvar_uw`, so the word means the same thing on both arms.
DISP_UW = 1.129


def main(argv):
    n_total = int(argv[1]) if len(argv) > 1 else 1000
    only = None
    if '--only' in argv:
        only = set(argv[argv.index('--only') + 1].split(','))
    emp, targets = empirical_targets()
    print('EMPIRICAL TARGET SIGNS, Spearman(visible modes, characteristic):')
    print(' ', {k: round(v, 3) for k, v in targets.items()})
    print(f'\ngenerating {len(CANDIDATES)} draft corpora at {n_total} datasets '
          'each. drafts only -- never a paper number.\n')

    rows, corr_rows = [], []
    for name, overrides in CANDIDATES.items():
        if only is not None and name not in only:
            continue
        cfg = corpus.scaled_config(G.DEFAULT, n_total,
                                   n_probe=max(5, n_total // 200))
        if overrides:
            cfg = dataclasses.replace(cfg, **overrides)
        t0 = time.time()
        # A corpus is never overwritten (decision 26), so an existing draft for
        # this candidate is REUSED rather than relabelled. That also makes the
        # script cheap to re-run when only one candidate is added.
        label = f'draft_joint_{name}'
        d = os.path.join(ROOT, 'data', 'processed', f'corpus_{label}')
        # A DRAFT IS REUSED ONLY IF IT WAS GENERATED UNDER THE CONFIGURATION
        # BEING ASKED FOR. It used to be reused whenever the directory existed,
        # and Stage 3 found `current` coming back from a draft made before the
        # Stage 2h regeneration -- `cv_log10_mean` 0.129 against the shipped
        # 0.329 and `min_q1_over_iqr` 0.5 against 0.2 -- so every candidate was
        # being compared against the SUPERSEDED configuration while the table
        # said `current`. A corpus is never overwritten (decision 26), so a
        # stale draft is reported and regenerated under a dated label rather
        # than replaced.
        if os.path.isdir(d) and _config_matches(d, cfg):
            print(f'  {name:24s} reusing existing draft', flush=True)
        else:
            if os.path.isdir(d):
                label = f'{label}_{time.strftime("%Y-%m-%d")}'
                d2 = os.path.join(ROOT, 'data', 'processed', f'corpus_{label}')
                print(f'  {name:24s} existing draft was generated under a '
                      f'DIFFERENT configuration; regenerating as {label}',
                      flush=True)
                if os.path.isdir(d2):
                    raise SystemExit(
                        f'{label} also exists; delete it or rename the '
                        f'candidate rather than overwriting a corpus')
            d = corpus.generate_corpus(cfg, label, progress=False)
        metrics, values, _meta = corpus.load_corpus(d)
        modes = visible_modes(values, metrics)
        m = metrics.merge(modes, left_on='dataset', right_index=True,
                          how='inner')
        row = dict(candidate=name, datasets=len(m),
                   seconds=round(time.time() - t0),
                   median_cv=m.coeffvar.median(), max_cv=m.coeffvar.max(),
                   share_cv_over_089=float((m.coeffvar > 0.889).mean()),
                   multimodal=float((m.modes_fitted >= 2).mean()),
                   dispersed=float((m.coeffvar > 0.889).mean()),
                   multimodal_and_dispersed=float(
                       ((m.modes_fitted >= 2) & (m.coeffvar > 0.889)).mean()),
                   multimodal_given_dispersed=float(
                       (m.modes_fitted >= 2)[m.coeffvar > 0.889].mean())
                   if (m.coeffvar > 0.889).any() else float('nan'))
        # THE WEIGHT-INVARIANT CELLS. `coeffvar` is measured under the corpus's
        # own market weights and `coeffvar_uw` is not, so only the second can be
        # compared with a real arm whose weight rule has changed. Both are
        # reported; the unweighted one is the one to judge on.
        if 'coeffvar_uw' in m.columns:
            du = m.coeffvar_uw > DISP_UW
            row.update(median_cv_uw=float(m.coeffvar_uw.median()),
                       dispersed_uw=float(du.mean()),
                       multimodal_and_dispersed_uw=float(
                           ((m.modes_fitted >= 2) & du).mean()),
                       multimodal_given_dispersed_uw=float(
                           (m.modes_fitted >= 2)[du].mean())
                       if du.any() else float('nan'))
        signs_right = 0
        for c in CHARS:
            if c not in m.columns:
                continue
            rho = spearmanr(m.modes_fitted, m[c])[0]
            corr_rows.append(dict(candidate=name, characteristic=c,
                                  synthetic=rho, empirical=targets.get(c),
                                  sign_matches=bool(
                                      np.sign(rho) == np.sign(targets.get(c, 0)))))
            signs_right += int(np.sign(rho) == np.sign(targets.get(c, 0)))
        row['signs_matching'] = f'{signs_right} of {len(CHARS)}'
        # The MARGINAL objective this project has always tuned on: mean
        # standardized Wasserstein distance between the two arms, one
        # characteristic at a time. Kept so a candidate that fixes the joint
        # structure by wrecking the margins is visible as such.
        cmp = coverage.distribution_comparison(emp, m)
        row['marginal_objective'] = float(cmp.w1_standardized.mean())
        row['coeffvar_margin'] = float(
            cmp.loc[cmp.metric == 'coeffvar', 'w1_standardized'].iloc[0])
        row['weighting_margin'] = float(
            cmp.loc[cmp.metric == 'w_v_uw_wasserstein',
                    'w1_standardized'].iloc[0])
        rows.append(row)
        print(f"  {name:20s} {row['seconds']:4d}s  multimodal "
              f"{100*row['multimodal']:5.1f} pct  "
              f"multimodal+dispersed {100*row['multimodal_and_dispersed']:5.2f} pct"
              f"  signs {row['signs_matching']}", flush=True)

    summary = pd.DataFrame(rows)
    corr = pd.DataFrame(corr_rows)
    print('\n\nSUMMARY (real arm: 31.5 pct multimodal, 16.2 pct multimodal AND '
          'dispersed, max CV 6.93)')
    print(summary.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print('\n\nSIGN OF Spearman(visible modes, characteristic) BY CANDIDATE')
    print(corr.pivot(index='characteristic', columns='candidate',
                     values='synthetic')
          .join(corr.drop_duplicates('characteristic')
                .set_index('characteristic')['empirical'])
          .to_string(float_format=lambda v: f'{v:+.3f}'))
    os.makedirs(OUT, exist_ok=True)
    summary.to_csv(os.path.join(OUT, 'TABLE_CorpusJointCandidates.csv'),
                   index=False)
    corr.to_csv(os.path.join(OUT, 'TABLE_CorpusJointCorrelations.csv'),
                index=False)
    print(f'\nwrote {OUT}/TABLE_CorpusJointCandidates.csv and '
          'TABLE_CorpusJointCorrelations.csv')


if __name__ == '__main__':
    main(sys.argv)
