"""Does the choice of UQ method change whether you EARN THE CREDIT? Stage 2h.

THE AUTHOR'S FRAMING, and it is a better one than the study currently uses.
Practitioners run LCAs to earn green-building certification points, and the
credit is written as a threshold: demonstrate a reduction of at least X. Under
a probabilistic LCA that claim would naturally become a threshold on a
CONFIDENCE -- "demonstrate a 10 percent reduction with 75 percent confidence".

WHAT THE STUDY ALREADY HAS, AND WHAT IT NEVER ASKED. `plca.reduction_statement`
already computes P(this intervention delivers at least 5, 10 or 20 percent of
the building), and the run against the true parents already scores the error in
that probability. What no stage has asked is the DECISION that probability is
used for: **is P at or above the confidence threshold, so the credit is
earned?** That is a different question from "how wrong is P", and it is the one
a practitioner experiences.

WHY IT IS DOUBLY FRAGILE, WHICH IS THE POINT. A credit decision inherits the
error in the probability AND a cliff at the threshold. Two methods that agree
about the probability to within a few points still disagree about the credit
whenever the true probability sits near the line. So the disagreement rate is
not a property of the methods alone; it is a property of where the design sits
relative to the threshold, and this reports both.

THREE THINGS IT MEASURES, for a grid of (reduction tier, confidence level):

    method disagreement   the share of cases where at least two of the six
                          methods land on opposite sides of the credit line
    error against truth   the share where a method says earned and the true
                          distributions say not, or the reverse
    where the cliff bites the same, split by how far the TRUE probability sits
                          from the threshold

NOTHING IS RE-RUN. Every number comes from the 60,000-row intervention table
the third notebook already wrote, which carries the probability, the true
probability and the error for all three tiers.

A SOURCING NOTE FOR THE MANUSCRIPT. The tiers here are the study's own 5, 10
and 20 percent, which happen to match the tiered structure certification
schemes use. **The exact wording, tier and confidence level of any specific
credit must be sourced before the paper cites one**; this project has had to
withdraw one figure quoted from memory already.

    conda run -n compareuq python audits/credit_threshold.py
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

import plca as PL            # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'audits')
OUT = os.path.join(ROOT, 'outputs', 'tables')

#: Reduction tiers the study computes, as percentages of the whole building.
TIERS = (5, 10, 20)

#: Confidence levels a credit might be written at. 0.5 is "more likely than
#: not", which is what a deterministic LCA implicitly claims.
CONFIDENCES = (0.50, 0.60, 0.75, 0.90, 0.95)

#: The two interventions the study models. `cap` specifies a better product;
#: `qty` uses less of the material.
STRATEGIES = ('cap', 'qty')


def credit_table(df, strategy, tier, confidence):
    """Per case: does each method earn the credit, and does the truth?"""
    col = f'{strategy}_p_reduction_over_{tier}'
    if col not in df.columns:
        return None
    work = df[['plca', 'dataset', 'method', col, f'{col}__truth']].copy()
    work = work.rename(columns={col: 'p', f'{col}__truth': 'p_truth'})
    work = work.dropna(subset=['p', 'p_truth'])
    work['earned'] = work.p >= confidence
    work['earned_truth'] = work.p_truth >= confidence
    return work


def summarize(work, strategy, tier, confidence):
    per_case = work.groupby(['plca', 'dataset'])
    # DISAGREEMENT BETWEEN METHODS: at least two of the six on opposite sides.
    disagree = per_case.earned.nunique() > 1
    # ERROR AGAINST THE TRUTH, per method.
    work = work.assign(wrong=work.earned != work.earned_truth)
    by_method = work.groupby('method').wrong.mean()
    truth_rate = per_case.earned_truth.first().mean()
    return dict(
        strategy=strategy, tier=tier, confidence=confidence,
        n_cases=int(per_case.ngroups),
        truth_earns=float(truth_rate),
        methods_disagree=float(disagree.mean()),
        worst_method_wrong=float(by_method.max()),
        best_method_wrong=float(by_method.min()),
        spread_across_methods=float(by_method.max() - by_method.min()),
        worst_method=str(by_method.idxmax()),
        best_method=str(by_method.idxmin()))


def near_the_line(work, confidence, bands=(0.02, 0.05, 0.10, 0.25, 1.01)):
    """Disagreement split by how far the TRUE probability sits from the line.

    This is what says the fragility is the CLIFF and not the methods: a design
    whose true confidence is nowhere near the threshold is decided the same way
    by everything.
    """
    per = work.groupby(['plca', 'dataset']).agg(
        disagree=('earned', lambda s: s.nunique() > 1),
        p_truth=('p_truth', 'first'))
    per['distance'] = (per.p_truth - confidence).abs()
    rows, lo = [], 0.0
    for hi in bands:
        pick = per[(per.distance >= lo) & (per.distance < hi)]
        if len(pick):
            rows.append(dict(from_line=f'{lo:.2f}-{hi:.2f}', n=len(pick),
                             disagree=float(pick.disagree.mean())))
        lo = hi
    return pd.DataFrame(rows)


def main():
    os.makedirs(TABLES, exist_ok=True)
    pd.set_option('display.width', 220)
    path = os.path.join(OUT, 'TABLE_PLCATruthIntervention.csv.gz')
    if not os.path.exists(path):
        print(f'{path} is missing; run the third notebook first.')
        return
    df = pd.read_csv(path)
    print(f'{len(df)} rows, {df.method.nunique()} methods, '
          f'{df.groupby(["plca","dataset"]).ngroups} cases')

    rows = []
    for strategy in STRATEGIES:
        for tier in TIERS:
            for conf in CONFIDENCES:
                work = credit_table(df, strategy, tier, conf)
                if work is None or work.empty:
                    continue
                rows.append(summarize(work, strategy, tier, conf))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TABLES, 'TABLE_CreditThreshold.csv'), index=False)

    print()
    print('=' * 78)
    print('DOES THE CHOICE OF UQ METHOD CHANGE WHETHER THE CREDIT IS EARNED?')
    print('=' * 78)
    print('`truth_earns` is the share of cases where the TRUE distributions')
    print('clear the bar, so a tier and confidence where it is near 0 or 1 is')
    print('one almost nobody is near and the disagreement there is rare by')
    print('construction rather than by the methods agreeing.')
    print()
    for strategy in STRATEGIES:
        s = out[out.strategy == strategy]
        if s.empty:
            continue
        print(f'--- {strategy} ---')
        print(s[['tier', 'confidence', 'truth_earns', 'methods_disagree',
                 'best_method_wrong', 'worst_method_wrong', 'worst_method']]
              .to_string(index=False, float_format=lambda v: f'{v:.4f}'))
        print()

    # THE CLIFF, at the author's own example.
    work = credit_table(df, 'cap', 10, 0.75)
    if work is not None and not work.empty:
        near = near_the_line(work, 0.75)
        near.to_csv(os.path.join(TABLES, 'TABLE_CreditThresholdCliff.csv'),
                    index=False)
        print('=' * 78)
        print('WHERE THE DISAGREEMENT COMES FROM: a 10 pct reduction at 75 pct')
        print('confidence, by how far the TRUE confidence sits from the line')
        print('=' * 78)
        print(near.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
        print()
        print('If disagreement is concentrated in the first row, the fragility')
        print('is the THRESHOLD and not the methods: a design comfortably over')
        print('or under the bar is called the same way by all six.')


if __name__ == '__main__':
    main()
