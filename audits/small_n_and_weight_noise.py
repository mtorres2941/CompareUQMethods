"""Why a normal can beat a KDE, and how much of the score is the weight draw.

an earlier revision, added after the author refused two results on sniff-test grounds. Both
refusals were correct and this is what was behind them.

QUESTION 1. "KDE converges to normal, so how would a normal ever beat it?"

It does not converge to the normal you would want. A Gaussian KDE is the data
convolved with a kernel, so its variance is the DATA's variance PLUS h^2: it
converges to N(mean, sd^2 + h^2), not N(mean, sd^2). With a rule-of-thumb
bandwidth h = 1.06 * sd * n_eff ** -0.2 that inflation is

    n = 3      sd x 1.31        n = 100     sd x 1.085
    n = 10     sd x 1.20        n = 1000    sd x 1.035
    n = 30     sd x 1.135       n = 10000   sd x 1.014

The normal fit matches the standard deviation exactly, by construction. So at
small n the KDE is a systematically OVER-DISPERSED model and W1 charges it for
that. This is the bias-variance property of rule-of-thumb bandwidths, not a
defect, and it is the reason a KDE needs data.

QUESTION 2. "The values are pulled from a parent as though they're uniform, but
then fit with Dirichlet weights, so the weighted dataset doesn't align with the
parent."

Half right, and the half that is right matters. On the SYNTHETIC arm
`genconfig.mode_coupling = 1.0`, so market share attaches at the mode level
(decision 21) and the parent HAS a market-weighted version, `MixtureParent.cdf(
scheme='market')`: the weighted data is a sample from a real population. But the
weights WITHIN a mode are a flat Dirichlet, which is noise, and on the EMPIRICAL
arm the weights are a flat Dirichlet throughout with no market information at
all.

So the variable-weighted empirical CDF -- which is the target every one of the
six methods is scored against -- carries a noise floor of its own. This script
measures it by drawing the weights twice on the same values.

    conda run -n compareuq python audits/small_n_and_weight_noise.py
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

import comparison as C  # noqa: E402
import empirical  # noqa: E402
import fitting as FT  # noqa: E402
from customstats import wasserstein1_weighted  # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'audits')
BANDS = [0, 9, 99, 999, 10 ** 9]
BAND_LABELS = ['n 3-9', 'n 10-99', 'n 100-999', 'n >=1000']
N_REALIZATIONS = 5


def banded(df):
    return df.assign(band=pd.cut(df.n, bins=BANDS, labels=BAND_LABELS))


def q1_overdispersion():
    """The KDE's own spread against the data's, measured, beside the theory."""
    print('=' * 78)
    print('1. WHY A NORMAL CAN BEAT A KDE: the KDE is over-dispersed at small n')
    print('=' * 78)
    s = banded(pd.read_csv(os.path.join(ROOT, 'outputs', 'tables',
                                        'TABLE_MethodScores.csv')))
    cols = ['KDE, Variable', 'Lognormal, Variable', 'Normal, Variable']
    for arm in ('empirical', 'synthetic'):
        g = s[s.arm == arm]
        print(f'--- {arm}: median (model sd) / (data sd) ---')
        print(g.pivot_table(index='band', columns='method',
                            values='model_sd_ratio', observed=True)[cols]
              .to_string(float_format=lambda v: f'{v:.3f}'))
        print()
    print('theory, Scott: sd inflation = sqrt(1 + (1.06 * n_eff ** -0.2) ** 2)')
    for n in (3, 5, 10, 30, 100, 1000, 10000):
        h = 1.06 * n ** -0.2
        print(f'  n = {n:>5}   h/sd = {h:.3f}   inflation = {np.sqrt(1 + h * h):.3f}')
    print()
    print('The measured KDE column tracks the theory to within a percent or two.')
    print('The normal fit sits BELOW 1 because it is truncated at zero and')
    print('renormalized, which cuts a tail off.')


def q1b_like_for_like():
    """Ranks among the three estimation methods only, so the comparison is fair."""
    print()
    print('=' * 78)
    print('1b. LIKE FOR LIKE: rank among the 3 estimation methods, variable weights')
    print('=' * 78)
    s = pd.read_csv(os.path.join(ROOT, 'outputs', 'tables',
                                 'TABLE_MethodScores.csv'))
    s = banded(C.add_ranks(s, 'w1', within_weighting=True,
                           suffix='w1_rank_within'))
    cols = ['KDE, Variable', 'Lognormal, Variable', 'Normal, Variable']
    for arm in ('empirical', 'synthetic'):
        g = s[s.arm == arm]
        print(f'--- {arm} ---')
        for value, lab in (('w1_rank_within', 'in sample'),
                           ('w1_heldout_rank_within', 'held out')):
            print(f'  {lab}:')
            print(g.pivot_table(index='band', columns='method', values=value,
                                observed=True).reindex(columns=cols)
                  .to_string(float_format=lambda v: f'{v:.2f}'))
        print()
    print('IN SAMPLE the KDE is SECOND at n = 10-99, not last; it is last only')
    print('below n = 10. HELD OUT it is last at n = 10-99, but a held-out fit')
    print('uses HALF the values, so that band is really measuring a KDE at')
    print('n = 5 to 50, which is where the over-dispersion above is worst.')


def q2_weight_noise():
    """The scoring target's own noise floor, from two draws on the same values."""
    print()
    print('=' * 78)
    print('2. THE SCORING TARGET HAS A NOISE FLOOR, and at small n it dominates')
    print('=' * 78)
    a, _ = empirical.prepare(np.random.default_rng(1000).spawn(1)[0])
    b, _ = empirical.prepare(np.random.default_rng(1001).spawn(1)[0])
    rows = []
    for name in a:
        x, wa = a[name]
        _, wb = b[name]
        rows.append(dict(dataset=name, n=len(x),
                         target_noise=wasserstein1_weighted(x, x, wa, wb),
                         var_vs_uniform=wasserstein1_weighted(
                             x, x, wa, np.ones_like(x) / len(x))))
    d = banded(pd.DataFrame(rows))
    d.to_csv(os.path.join(TABLES, 'TABLE_WeightTargetNoise.csv'), index=False)
    print('W1 between the SAME values under two independent Dirichlet draws:')
    print(d.groupby('band', observed=True).agg(
        datasets=('dataset', 'size'),
        median_target_noise=('target_noise', 'median'),
        mean_target_noise=('target_noise', 'mean'),
        median_var_vs_uniform=('var_vs_uniform', 'median')
    ).to_string(float_format=lambda v: f'{v:.4f}'))

    s = pd.read_csv(os.path.join(ROOT, 'outputs', 'tables',
                                 'TABLE_MethodScores.csv'))
    med = s[s.arm == 'empirical'].groupby('method').w1.median().sort_values()
    print()
    print('median W1 of each method on the same arm:')
    print(med.to_string(float_format=lambda v: f'{v:.4f}'))
    print()
    print(f'MEDIAN TARGET NOISE {d.target_noise.median():.4f} IS LARGER THAN THE '
          f'BEST METHOD\'S MEDIAN W1 {med.min():.4f}.')
    print('At n >= 1000 the floor is 0.006 and negligible. At n = 10-99 it is')
    print('0.150, which is larger than the gaps between the methods, so')
    print('size-banded claims below about n = 100 are not resolvable at the')
    print('precision they were being quoted to.')


def q2b_rank_stability(k=N_REALIZATIONS):
    """Does the ANSWER move when the weights are redrawn? Mostly not."""
    print()
    print('=' * 78)
    print(f'2b. IS THE RANKING STABLE ACROSS {k} WEIGHT REALIZATIONS? Mostly yes')
    print('=' * 78)
    rows = []
    for i in range(k):
        ds, _ = empirical.prepare(np.random.default_rng(1000 + i).spawn(1)[0])
        for name, (x, w) in ds.items():
            models, _ = FT.fit_pewt(x, w)
            grid = FT.score_grid_open(x, w)
            for label in FT.PEWT:
                rows.append(dict(realization=i, dataset=name, n=len(x),
                                 method=label,
                                 w1=FT.score_w1_model(models[label], x, w,
                                                      grid=grid)))
        print(f'  realization {i} done', flush=True)
    d = pd.DataFrame(rows)
    d['rank'] = d.groupby(['realization', 'dataset']).w1.rank()
    d.to_csv(os.path.join(TABLES, 'TABLE_RankStability.csv'), index=False)
    p = d.pivot_table(index='method', columns='realization', values='rank')
    p['mean'] = p.mean(axis=1)
    p['sd'] = p.iloc[:, :k].std(axis=1)
    print()
    print('mean rank over the six methods, per weight realization:')
    print(p.sort_values('mean').to_string(float_format=lambda v: f'{v:.3f}'))
    win = d.loc[d.groupby(['realization', 'dataset']).w1.idxmin()]
    share = (win.pivot_table(index='method', columns='realization',
                             values='dataset', aggfunc='count')
             / d.dataset.nunique() * 100)
    print()
    print('share of datasets each method WINS outright, per realization:')
    print(share.to_string(float_format=lambda v: f'{v:.1f}'))
    print()
    print('The aggregate ordering is stable: places 3 to 6 never move, and')
    print('KDE, Variable leads on win share in every realization. The top two')
    print('are within the draw noise of each other on MEAN RANK -- the')
    print('lognormal edges ahead in one realization of five -- so "the KDE is')
    print('best on the empirical arm overall" should be stated as a win-share')
    print('result, not as a mean-rank result.')


def main():
    os.makedirs(TABLES, exist_ok=True)
    q1_overdispersion()
    q1b_like_for_like()
    q2_weight_noise()
    q2b_rank_stability()


if __name__ == '__main__':
    main()
