"""Judgment-driven models on the same axis as data-driven ones. Stage 2h.

Every method this study compares turns a SET of declarations into a
distribution. The probabilistic LCA methods most readers actually use do not:
the pedigree matrix is formulaic expert judgment applied where data is absent,
and a uniform or a triangular over a plausible range is what someone reaches
for with two or three numbers and no dataset.

    THEY CANNOT BE COMPARED LIKE FOR LIKE AND THIS DOES NOT CLAIM THEY CAN.

What makes the comparison possible is the yardstick, not a claim about the
methods: Stage 2e's error against the TRUE distribution the data was drawn
from does not care how a model was built. Decision 124.

    TWO DIMENSIONS, AND THE SECOND IS THE ONE THAT MATTERS.

A pedigree model is a SPREAD around a POINT ESTIMATE the practitioner already
holds, and nothing puts that point where the category's mean is. This project
has twice found that BIAS adds across a building's materials while random error
cancels, so a judgment model with a well-chosen spread and a displaced center
fails the way the normal fit does. Sweeping only the spread would miss that and
would flatter the pedigree approach.

    THE PRIMARY LOCATION MODEL HAS NO FREE PARAMETER: one declaration, drawn at
    random, which is what a practitioner without a dataset holds. The offset
    distribution then falls out of the data instead of being chosen. A
    deliberate offset is swept beside it so the sensitivity is mapped as well
    as sampled, and the two are reported JOINTLY.

    THE DELIVERABLE is one sentence: a judgment-driven model whose spread is
    within X of the data's own, and whose center is within Y of the truth, does
    not change the answer; beyond that it does.

    conda run -n compareuq python audits/judgment_arm.py [n_datasets] [n_pairs]
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

import corpus                       # noqa: E402
import fitting as FT                # noqa: E402
import judgment as JD               # noqa: E402
import plca as PL                   # noqa: E402
import recovery as R                # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'audits')

#: The joint grid. `None` in the offset column is the PRIMARY, no-free-
#: parameter location model: the center is one declaration drawn at random,
#: which is what a practitioner without a dataset holds.
#:
#: THE MODE MATTERS MORE THAN THE SIZE, which the first run of this script
#: found. A displacement applied identically to every material cancels EXACTLY
#: in a design comparison, because both options' totals scale by the same
#: factor. So a sweep over a common offset alone reports a null that is an
#: artifact of the sweep. Both modes are therefore swept.
GSD_RATIOS = JD.GSD_RATIOS
OFFSET_CELLS = ([(None, 'oneEPD')]
                + [(f, 'common') for f in JD.CENTER_OFFSETS if f != 0.0]
                + [(0.0, 'common')]
                + [(f, 'independent') for f in JD.CENTER_OFFSETS if f > 0.0])


def cell_name(ratio, offset, mode):
    off = 'oneEPD' if offset is None else f'{mode[:3]}{offset:+.2f}'
    return f'judgment gsd x{ratio:g} center {off}'


# ---------------------------------------------------------------------------
# 1. the fit level: how far a judgment model sits from the true distribution
# ---------------------------------------------------------------------------
def fit_level(n_datasets, rng):
    met, values, _ = corpus.load_corpus(with_values=True)
    met = met[met.dataset.astype(str).str.startswith('dataset')]
    pick = met.sample(min(n_datasets, len(met)), random_state=0)
    vals = corpus.as_dict(values)
    specs = corpus.load_parent_objects(
        datasets=sorted(str(d) for d in pick.dataset))

    rows, t0 = [], time.time()
    for i, (_, r) in enumerate(pick.iterrows()):
        ds = str(r.dataset)
        if ds not in vals or ds not in specs:
            continue
        x, w = np.asarray(vals[ds][0]), np.asarray(vals[ds][1])
        if len(x) < 3:
            continue
        parent = specs[ds]
        grid = R.recovery_grid(x, w, parent)
        gsd_data = JD.data_gsd(x)
        if not np.isfinite(gsd_data) or gsd_data <= 1.0:
            continue
        base = dict(dataset=ds, n=len(x), gsd_data=gsd_data)
        # the six data-driven methods, as the reference
        models, _ = FT.fit_pewt(x, w)
        for m, mod in models.items():
            rows.append(dict(base, arm='data-driven', method=m,
                             gsd_ratio=np.nan, offset=np.nan,
                             w1=R.w1_against_parent(mod, parent, 'market',
                                                    grid)))
        for ratio in GSD_RATIOS:
            gsd = 1.0 + (gsd_data - 1.0) * ratio
            for off, mode in OFFSET_CELLS:
                mods = JD.judgment_models(x, rng, gsd=gsd, offset_frac=off,
                                          offset_mode=mode)
                for shape, mod in mods.items():
                    rows.append(dict(
                        base, arm='judgment', method=shape, gsd_ratio=ratio,
                        offset=(np.nan if off is None else off),
                        offset_mode=mode,
                        center=('oneEPD' if off is None
                                else f'{mode[:3]}{off:+.2f}'),
                        w1=R.w1_against_parent(mod, parent, 'market', grid)))
        if (i + 1) % 100 == 0:
            print(f'  fit level {i+1}/{len(pick)}  {time.time()-t0:.0f}s',
                  flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. the decision level: the design comparison, which is what a designer does
# ---------------------------------------------------------------------------
def decision_level(n_pairs, rng, nmats=4, neccs=2000):
    """The design comparison with judgment models beside the data-driven six.

    A NULL HERE WOULD BE THE MOST USEFUL RESULT IN THE STAGE, because it would
    say the practice most readers use is also safe for the decision they make.
    """
    met, values, _ = corpus.load_corpus(with_values=True)
    met = met[met.dataset.astype(str).str.startswith('dataset')]
    groups = PL.resample_groups(sorted(met.dataset.astype(str)), nmats + 1,
                                n_pairs, rng)
    needed = sorted({d for g in groups for d in g})
    vals = corpus.as_dict(values)
    samplers = PL.LazySamplers(corpus.load_parent_objects(datasets=needed),
                               scheme=PL.TRUTH_SCHEME)

    models = {}
    for d in needed:
        x, w = np.asarray(vals[d][0]), np.asarray(vals[d][1])
        fitted, _ = FT.fit_pewt(x, w)
        gsd_data = JD.data_gsd(x)
        if not np.isfinite(gsd_data) or gsd_data <= 1.0:
            gsd_data = 1.2
        for ratio in GSD_RATIOS:
            gsd = 1.0 + (gsd_data - 1.0) * ratio
            for off, mode in OFFSET_CELLS:
                mods = JD.judgment_models(x, rng, gsd=gsd, offset_frac=off,
                                          offset_mode=mode)
                nm = cell_name(ratio, off, mode)
                fitted[f'{nm} | pedigree'] = mods['pedigree']
                if ratio == 1.0:
                    # The other two judgment shapes only at the matched spread,
                    # which keeps the run affordable and asks the question they
                    # are actually for: given the same two judgment inputs, does
                    # the SHAPE a practitioner assumes matter.
                    fitted[f'{nm} | uniform'] = mods['uniform']
                    fitted[f'{nm} | triangular'] = mods['triangular']
        models[d] = fitted
    methods = list(next(iter(models.values())))
    print(f'  {len(methods)} methods x {n_pairs} pairs', flush=True)
    return PL.swap_run(models, groups, rng, neccs=neccs, methods=methods,
                       samplers=samplers)


def main(n_datasets=500, n_pairs=250):
    os.makedirs(TABLES, exist_ok=True)
    pd.set_option('display.width', 220)
    rng = np.random.default_rng(20260924)

    print('=' * 78)
    print('1. FIT LEVEL: distance from the TRUE distribution')
    print('=' * 78)
    fit = fit_level(n_datasets, rng)
    fit.to_csv(os.path.join(TABLES, 'TABLE_JudgmentFitLevel.csv.gz'),
               index=False)
    report_fit(fit)

    print()
    print('=' * 78)
    print('2. DECISION LEVEL: the design comparison')
    print('=' * 78)
    dec = decision_level(n_pairs, rng)
    dec.to_csv(os.path.join(TABLES, 'TABLE_JudgmentDesignSwap.csv.gz'),
               index=False)
    report_decision(dec)
    return fit, dec


def report_fit(fit):
    dd = fit[fit.arm == 'data-driven']
    print()
    print('The six data-driven methods, median W1 against the market parent:')
    print(dd.groupby('method').w1.median()
          .sort_values().to_string(float_format=lambda v: f'{v:.4f}'))
    best = float(dd.groupby('method').w1.median().min())
    print(f'\nBest data-driven method: {best:.4f}. Every judgment cell below is')
    print('a MULTIPLE of that, so 1.0 would be as good as the best fit.')
    print()
    jd = fit[fit.arm == 'judgment']
    for shape in ('pedigree', 'uniform', 'triangular'):
        s = jd[jd.method == shape]
        if s.empty:
            continue
        piv = s.pivot_table(index='gsd_ratio', columns='center', values='w1',
                            aggfunc='median', dropna=False)
        cols = ['oneEPD'] + [c for c in piv.columns if c != 'oneEPD']
        print(f'--- {shape} --- as a MULTIPLE of the best data-driven method.')
        print('    `oneEPD` is the primary model: the center is one random')
        print('    declaration. `com` is a displacement applied to every')
        print('    material alike, `ind` one drawn per material.')
        print((piv[cols] / best).to_string(float_format=lambda v: f'{v:.2f}'))
        print()


def report_decision(dec):
    print()
    d = dec.copy()
    d['abs_err'] = d.discernibility__error.abs()
    data_driven = [m for m in d.method.unique() if '|' not in m]
    print('P(option B beats A): error against the truth, PER PAIR, which is')
    print('what ONE design decision carries. Lower is better.')
    print()
    ref = d[d.method.isin(data_driven)].groupby('method').abs_err.mean()
    print('the six data-driven methods:')
    print(ref.sort_values().to_string(float_format=lambda v: f'{v:.4f}'))
    worst_dd = float(ref.max())
    print()
    j = d[~d.method.isin(data_driven)].copy()
    if j.empty:
        return
    parts = j.method.str.split(' \\| ', regex=True, expand=True)
    j['cell'], j['shape'] = parts[0], parts[1]
    j['gsd_ratio'] = j.cell.str.extract(r'gsd x([0-9.]+)').astype(float)
    j['center'] = j.cell.str.extract(r'center (\S+)')
    j['mode'] = np.where(j.center.str.startswith('ind'), 'independent',
                         np.where(j.center == 'oneEPD', 'oneEPD', 'common'))
    for shape in ('pedigree', 'uniform', 'triangular'):
        s = j[j['shape'] == shape]
        if s.empty:
            continue
        piv = s.pivot_table(index='gsd_ratio', columns='center',
                            values='abs_err', aggfunc='mean')
        print(f'--- {shape} ---   (worst data-driven method: {worst_dd:.4f})')
        print(piv.to_string(float_format=lambda v: f'{v:.4f}'))
        print()
    print('A judgment cell whose error is at or below the worst data-driven')
    print('method is one a practitioner could use for THIS decision without')
    print('being worse off than a defensible data-driven choice.')


if __name__ == '__main__':
    a = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    b = int(sys.argv[2]) if len(sys.argv) > 2 else 250
    main(a, b)
