"""Stage 2a, Part 0 item 1a: confirm the seeding-collapse signature directly.

Runs the frozen pre-Stage-1 generator and shows, rather than asserts, that
(a) the three shape parameters are constants, and (b) component draws are a
deterministic function of (type, count).

Writes: outputs/tables/stage2a/TABLE_2a_CollapseSignature.csv
"""
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../../src'))

import legacy_generator_pre_stage1 as legacy
import datageneration as current

rows = []

# --- (a) the shape parameters are constants under the legacy code -----------
# Each is the first uniform draw from a fresh default_rng(0), so it is the same
# number every call, for every dataset, in the entire study.
r = np.random.default_rng(0)
a_const = float(r.uniform(-1, 6))
r = np.random.default_rng(0)
df_const = float(r.uniform(0.5, 5.0))
r = np.random.default_rng(0)
s_const = float(r.uniform(0.5, 1.5))
rows.append(dict(check='shape_param_skewnorm_a', legacy_value=a_const,
                 claimed=3.458732, note='constant for all skewnorm components, all datasets'))
rows.append(dict(check='shape_param_studentt_df', legacy_value=df_const,
                 claimed=3.366328, note='constant for all studentt components, all datasets'))
rows.append(dict(check='shape_param_lognorm_s', legacy_value=s_const,
                 claimed=1.136962, note='constant for all lognorm components, all datasets'))

# --- (b) component draws are deterministic given (type, count) -------------
for t in ['gauss', 'skewnorm', 'studentt', 'lognorm']:
    x1 = legacy.generate_random_numbers(t, 10.0, 1.0, 50)
    x2 = legacy.generate_random_numbers(t, 10.0, 1.0, 50)
    maxdiff = float(np.max(np.abs(x1 - x2)))
    rows.append(dict(check=f'legacy_repeat_identical_{t}', legacy_value=maxdiff,
                     claimed=0.0,
                     note='max abs difference between two calls with the same (type, loc, scale, count)'))

# Two gauss components of equal length but different loc/scale: exact affine images.
g1 = legacy.generate_random_numbers('gauss', 5.0, 0.3, 200)
g2 = legacy.generate_random_numbers('gauss', 19.0, 1.4, 200)
z1 = (g1 - g1.mean()) / g1.std()
z2 = (g2 - g2.mean()) / g2.std()
rows.append(dict(check='legacy_two_gauss_standardized_maxdiff',
                 legacy_value=float(np.max(np.abs(z1 - z2))), claimed=0.0,
                 note='two gauss components of equal length are exact affine images'))

# --- the same three checks under the current (Stage 1) generator ------------
rng = np.random.default_rng(20260911)
for t in ['gauss', 'skewnorm', 'studentt', 'lognorm']:
    x1 = current.generate_random_numbers(t, 10.0, 1.0, 50, rng)
    x2 = current.generate_random_numbers(t, 10.0, 1.0, 50, rng)
    rows.append(dict(check=f'current_repeat_identical_{t}',
                     legacy_value=float(np.max(np.abs(x1 - x2))), claimed=np.nan,
                     note='same call twice under the corrected generator'))

g1 = current.generate_random_numbers('gauss', 5.0, 0.3, 200, rng)
g2 = current.generate_random_numbers('gauss', 19.0, 1.4, 200, rng)
z1 = (g1 - g1.mean()) / g1.std()
z2 = (g2 - g2.mean()) / g2.std()
rows.append(dict(check='current_two_gauss_standardized_maxdiff',
                 legacy_value=float(np.max(np.abs(z1 - z2))), claimed=np.nan,
                 note='Stage 1 handoff reports 3.36 here'))

# --- realized shape-parameter spread under the current generator -----------
rng = np.random.default_rng(20260911)
for name, lo, hi in [('skewnorm_a', -1, 6), ('studentt_df', 0.5, 5.0), ('lognorm_s', 0.5, 1.5)]:
    draws = rng.uniform(lo, hi, 10000)
    rows.append(dict(check=f'current_shape_spread_{name}', legacy_value=float(draws.std()),
                     claimed=0.0, note=f'sd of the shape parameter across draws; legacy sd is exactly 0'))

df = pd.DataFrame(rows)
os.makedirs('../../outputs/tables/stage2a', exist_ok=True)
df.to_csv('../../outputs/tables/stage2a/TABLE_2a_CollapseSignature.csv', index=False)
pd.set_option('display.width', 200, 'display.max_colwidth', 70)
print(df.to_string(index=False))
