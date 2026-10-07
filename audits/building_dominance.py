"""How far the biggest material leads the next, in 292 REAL buildings.

WHY THIS EXISTS. The safe-lead rule says a contribution ranking is only safe to
report once the leading material's mean contribution exceeds the next by about
2.3 times. Until now the paper could anchor that against exactly ONE real
building element -- the Concrete-Precast staircase of Marsh, Lewis, Hattam and
Allen (in press), at a top-two ratio of 1.02 -- which is an anecdote rather than
a distribution.

THE INPUT IS EXTERNAL AND GITIGNORED, so this script freezes what it derives.
`refs/28462145/` holds the Benke et al. (2025) harmonized whole-building LCA
dataset, which is 100 MB and lives under `refs/` by decision 1. This script
reads it, reduces it to one row per building, and writes that into
`data/raw/building_top2_benke2025.csv`, which IS tracked -- the same pattern
decision 31 used for the EC3 extract, and for the same reason: the figure must
be reproducible from a clean clone that does not have the 100 MB input.

    Benke, B., Chafart, M., Shen, Y., Ashtiani, M., Carlisle, S., Simonen, K.
    (2025). A Harmonized Dataset of High-Resolution Whole Building Life Cycle
    Assessment Results in North America: Data only - First Public Release.
    figshare. https://doi.org/10.6084/m9.figshare.28462145.v1

    conda run -n compareuq python audits/building_dominance.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
SRC = os.path.join(ROOT, 'refs', '28462145', 'full_lca_results.xlsx')
FROZEN = os.path.join(ROOT, 'data', 'raw', 'building_top2_benke2025.csv')
# THE WORKED EXAMPLE'S BUILDING, frozen the same way: one row per material type
# with its A1-A3 emissions AND its mass, so the case study in notebook 3 runs
# from a clean clone. Both columns are kept because the author's objection was
# that mass alone says nothing about contribution: Benke's own emission factor
# per kilogram is 1.91 for rebar against 0.28 to 0.37 for the concrete.
CASE_PROJECT = 138
FROZEN_CASE = os.path.join(ROOT, 'data', 'raw', 'building138_benke2025.csv')
OUT = os.path.join(ROOT, 'outputs', 'tables', 'audits')

#: A1-A3 only. The study is about embodied carbon COEFFICIENTS, which are
#: product-stage quantities; including use-stage replacement or end-of-life
#: would mix in service-life assumptions that have nothing to do with the
#: uncertainty method being tested.
STAGE = 'A1-A3'


def derive(path=SRC):
    """One row per building: how far its largest material leads the next."""
    d = pd.read_excel(path, usecols=['project_index', 'mat_type',
                                     'life_cycle_stage', 'gwp'])
    a = d[(d.life_cycle_stage == STAGE) & (d.gwp > 0)]
    g = a.groupby(['project_index', 'mat_type']).gwp.sum().reset_index()
    rows = []
    for project, x in g.groupby('project_index'):
        v = np.sort(x.gwp.to_numpy(float))[::-1]
        if len(v) < 2:
            continue
        rows.append(dict(project_index=int(project), n_materials=len(v),
                         top2_ratio=float(v[0] / v[1]),
                         top_share=float(v[0] / v.sum()),
                         top_material=str(x.loc[x.gwp.idxmax(), 'mat_type'])))
    return pd.DataFrame(rows).sort_values('top2_ratio').reset_index(drop=True)


def derive_case(path=SRC, project=CASE_PROJECT):
    """One building's materials: A1-A3 emissions, mass and intensity."""
    d = pd.read_excel(path, usecols=['project_index', 'mat_group', 'mat_type',
                                     'life_cycle_stage', 'inv_mass', 'gwp',
                                     'mui_gfa'])
    b = d[(d.project_index == project) & (d.life_cycle_stage == STAGE)]
    g = (b.groupby(['mat_group', 'mat_type'])
         .agg(mass_kg=('inv_mass', 'sum'), gwp_kgco2e=('gwp', 'sum'),
              mass_kg_per_m2=('mui_gfa', 'sum'))
         .reset_index().sort_values('gwp_kgco2e', ascending=False))
    g['share_of_gwp'] = g.gwp_kgco2e / g.gwp_kgco2e.sum()
    g['gwp_per_kg'] = g.gwp_kgco2e / g.mass_kg
    meta = pd.read_excel(os.path.join(os.path.dirname(path),
                                      'buildings_metadata.xlsx'))
    gfa = float(meta.loc[meta.project_index == project, 'bldg_gfa'].iloc[0])
    g['gwp_kgco2e_per_m2'] = g.gwp_kgco2e / gfa
    g.insert(0, 'project_index', project)
    g['bldg_gfa_m2'] = gfa
    return g.reset_index(drop=True)


def main():
    if not os.path.exists(SRC):
        print(f'{SRC} not found. It is gitignored by decision 1; the frozen '
              f'derivative at {FROZEN} is what the notebook reads.')
        return 1
    r = derive()
    os.makedirs(os.path.dirname(FROZEN), exist_ok=True)
    r.to_csv(FROZEN, index=False)
    derive_case().to_csv(FROZEN_CASE, index=False)
    os.makedirs(OUT, exist_ok=True)

    # The two thresholds the study publishes, at its own four materials.
    sys.path.insert(0, os.path.join(ROOT, 'src'))
    print(f'{len(r)} buildings, {STAGE}, materials at mat_type level')
    print(f'  materials per building: median {r.n_materials.median():.0f}, '
          f'range {r.n_materials.min()} to {r.n_materials.max()}')
    print('\nTOP-TWO CONTRIBUTION RATIO')
    for q in (0.10, 0.25, 0.50, 0.75, 0.90):
        print(f'   p{int(q * 100):<3d} {r.top2_ratio.quantile(q):.2f}')
    print(f'   min {r.top2_ratio.min():.2f}   max {r.top2_ratio.max():.2f}')
    print()
    for thr, lab in ((2.28, '1 pct chance the UQ method changes the leader'),
                     (1.73, '5 pct chance')):
        n = int((r.top2_ratio < thr).sum())
        print(f'   below {thr:.2f}x ({lab}): {100 * n / len(r):.0f} pct '
              f'({n} of {len(r)})')
    print('\n   Marsh et al. (in press) staircase anchor: 1.02')
    print('\nTHE THRESHOLD IS CALIBRATED AT FOUR MATERIALS AND THESE BUILDINGS '
          'HOLD A MEDIAN OF 37,')
    print('so if anything it UNDERSTATES the share at risk: decision 107 '
          'measures the 1 percent')
    print('crossing rising from 1.90 at two materials to 2.34 at twelve.')
    print(f'\nwrote {FROZEN} and {FROZEN_CASE}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
