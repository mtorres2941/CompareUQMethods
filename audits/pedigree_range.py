"""What geometric standard deviation can the pedigree matrix actually produce?

**WHY THIS EXISTS.** Stage 2h swept a judgment-driven model's spread RELATIVE
to the data's own, from half to three times it, because the pedigree matrix's
own table of uncertainty factors was not among this project's reference
materials and decision 49's amendment is a standing warning about quoting a
figure from memory. The author has since supplied Muller et al. (2016), so the
sweep can be anchored rather than left relative.

**AND THE FIRST READING OF IT WAS WRONG, WHICH IS THE REASON THIS IS A SCRIPT
AND NOT A CONSTANT.** A note taken from that paper recorded "GSD 1.279 basic
rising to 1.690 at scores 5,5,5,5,5". Those two numbers are neither GSDs nor a
range: 1.26 and 1.69 are the posterior uncertainty factors for ONE indicator,
the further technological correlation, at scores 2 and 3, for the manufacturing
sector (their Table 5). Reading them as a total spread understates the top of
the range and misplaces the bottom of it.

**THE ARITHMETIC, which is the part a reviewer checks.** Every factor in the
pedigree matrix is a contributor to the SQUARE of the geometric standard
deviation, which the paper states in as many words. They combine in log space,

    sigma_95 = sqrt( sum_i [ln(UF_i)] ** 2 )     over the five indicators
                                                 plus the basic uncertainty
    GSD_squared = exp(sigma_95)
    GSD         = exp(sigma_95 / 2)

so a model quoted as a GSD must halve the exponent, and quoting the combined
factor AS a GSD would double the spread.

This enumerates all 5 ** 5 score combinations and reports the range, so the
judgment arm's sweep can be placed on it.

    conda run -n compareuq python audits/pedigree_range.py
"""
import itertools
import os
import sys
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, HERE)

import empirical                     # noqa: E402
import judgment                      # noqa: E402
import tune_configuration as TC      # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'audits')

#: Ecoinvent's published pedigree uncertainty factors, scores 2 to 5; score 1
#: contributes a factor of exactly 1. Read from the "Prior" column of Table 3
#: of Muller, Lesage, Ciroth, Mutel, Weidema and Samson (2016), "Giving a
#: scientific basis for uncertainty factors used in global life cycle
#: inventory databases", Int J Life Cycle Assess 21:1185-1196, where "prior"
#: means the value ecoinvent uses and which that paper sets out to update.
ECOINVENT_UF = {
    'reliability':               (1.00, 1.05, 1.10, 1.20, 1.50),
    'completeness':              (1.00, 1.02, 1.05, 1.10, 1.20),
    'temporal correlation':      (1.00, 1.03, 1.10, 1.20, 1.50),
    'geographical correlation':  (1.00, 1.01, 1.02, 1.05, 1.10),
    'further technological':     (1.00, 1.05, 1.20, 1.50, 2.00),
}

#: The basic uncertainty factor for the row a building material falls in,
#: "Thermal energy, electricity, semi-finished products, material, waste",
#: from Table 4 of the same paper. It is 1.05 under every one of the three
#: sectors that table reports, so no sector choice is being made here.
BASIC_UF = 1.05


def gsd(scores, basic=BASIC_UF, table=ECOINVENT_UF):
    """The geometric standard deviation implied by five pedigree scores."""
    s2 = np.log(basic) ** 2
    for (name, ufs), score in zip(table.items(), scores):
        s2 += np.log(ufs[score - 1]) ** 2
    return float(np.exp(np.sqrt(s2) / 2.0))


def main():
    os.makedirs(TABLES, exist_ok=True)
    pd.set_option('display.width', 200)

    combos = list(itertools.product(range(1, 6), repeat=5))
    vals = np.array([gsd(c) for c in combos])
    df = pd.DataFrame(combos, columns=list(ECOINVENT_UF))
    df['gsd'] = vals
    df['gsd_squared'] = vals ** 2

    print('=' * 78)
    print('WHAT THE PEDIGREE MATRIX CAN PRODUCE, all 3,125 score combinations')
    print('=' * 78)
    print(f'  best  (1,1,1,1,1)  GSD {gsd((1,) * 5):.4f}   '
          f'GSD^2 {gsd((1,) * 5) ** 2:.4f}')
    print(f'  worst (5,5,5,5,5)  GSD {gsd((5,) * 5):.4f}   '
          f'GSD^2 {gsd((5,) * 5) ** 2:.4f}')
    print(f'  median             GSD {np.median(vals):.4f}')
    print(f'  quartiles          GSD {np.percentile(vals, 25):.4f} '
          f'to {np.percentile(vals, 75):.4f}')
    print()
    print('  ONE INDICATOR AT A TIME, the rest held at 1, so a reader can see')
    print('  which indicator carries the spread:')
    for name in ECOINVENT_UF:
        row = []
        for sc in range(1, 6):
            s = tuple(sc if n == name else 1 for n in ECOINVENT_UF)
            row.append(f'{gsd(s):.4f}')
        print(f'    {name:26s} ' + '  '.join(row))

    print()
    print('AGAINST THE DATA THE STUDY ACTUALLY HOLDS')
    ds, _ = empirical.prepare(np.random.default_rng(TC.SEED).spawn(1)[0])
    data = np.array([judgment.data_gsd(x, w) for x, w in ds.values()])
    data = data[np.isfinite(data)]
    print(f'  {len(data)} real categories, their own GSD:')
    print(f'    median {np.median(data):.4f}   quartiles '
          f'{np.percentile(data, 25):.4f} to {np.percentile(data, 75):.4f}'
          f'   range {data.min():.4f} to {data.max():.4f}')
    lo, hi = gsd((1,) * 5), gsd((5,) * 5)

    # THE RATIO MUST BE TAKEN THE WAY THE SWEEP TAKES IT, which is on the
    # EXCESS over 1 and not on the geometric standard deviation itself:
    # `audits/judgment_arm.py` sets `gsd = 1 + (gsd_data - 1) * ratio`. That is
    # the sound definition, because a GSD of 1 is no spread at all, so the
    # dispersion being scaled is `gsd - 1`; a straight ratio would also send a
    # tight category below 1, which is not a distribution. Reporting the
    # straight ratio against a sweep that uses the excess would misplace the
    # reachable range by a wide margin, so both are printed and the one the
    # sweep uses is named.
    def excess_ratio(model_gsd, data_gsd):
        return (model_gsd - 1.0) / (data_gsd - 1.0)

    med = float(np.median(data))
    print()
    print('  THE RATIO THE JUDGMENT ARM SWEEPS: (model GSD - 1) over')
    print('  (data GSD - 1), which is what `gsd = 1 + (gsd_data - 1) * ratio`')
    print('  inverts to')
    print(f'    against the MEDIAN category   '
          f'{excess_ratio(lo, med):.3f} to {excess_ratio(hi, med):.3f}')
    print(f'    against the WIDEST category   '
          f'{excess_ratio(lo, data.max()):.4f} to '
          f'{excess_ratio(hi, data.max()):.4f}')
    print(f'    against the TIGHTEST category '
          f'{excess_ratio(lo, data.min()):.3f} to '
          f'{excess_ratio(hi, data.min()):.1f}')
    print('  the STRAIGHT ratio, model GSD over data GSD, for contrast only:')
    print(f'    against the MEDIAN category   {lo / med:.3f} to {hi / med:.3f}')
    print()
    print(f'  swept in src/judgment.py: {judgment.GSD_RATIOS}')
    med_reach = [r for r in judgment.GSD_RATIOS
                 if excess_ratio(lo, med) <= r <= excess_ratio(hi, med)]
    print(f'  reachable on the MEDIAN category: {tuple(med_reach)}')
    # Per category, the largest swept ratio the matrix can reach at all.
    top = np.array([excess_ratio(hi, g) for g in data])
    for r in judgment.GSD_RATIOS:
        print(f'    ratio {r:<4} is reachable on '
              f'{float((top >= r).mean()) * 100:5.1f} pct of real categories')
    print()
    print(f'  share of real categories WIDER than the worst pedigree score: '
          f'{float((data > hi).mean()):.3f}')

    path = os.path.join(TABLES, 'TABLE_PedigreeRange.csv')
    df.to_csv(path, index=False)
    print()
    print(f'written to {os.path.relpath(path, ROOT)}')


if __name__ == '__main__':
    main()
