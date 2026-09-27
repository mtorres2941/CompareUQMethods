# Manuscript discrepancy log

Working checklist for revising
`outputs/manuscript/2026-09-10_Manuscript_CompareUQMethods.docx`. Every stage
appends to this file; nothing is deleted, entries are marked resolved.

Each entry gives what the manuscript says, what the code does, and whether the
fix belongs in the text or in the analysis.

Quotations are from the manuscript as extracted on 2026-09-11 (345 paragraphs,
108,164 characters, 98 unresolved advisor comments). Code references are to
commit `68ac3e2` on branch `stage-0-1-refactor`.

**Three entries seeded from the Stage 1 prompt were found to be wrong when
checked against the manuscript text.** They are kept below, marked CORRECTED,
because the corrected version is narrower and more useful than the original
claim. See entries 3, 4 and 6.

---

## 1. Monte Carlo sample size

| | |
|---|---|
| **Manuscript** | "Monte Carlo simulation (n = 10,000) was employed to sample ECCs from a probabilistic model". Repeated throughout the supplementary result definitions: "For each of 10,000 iterations, each of the four PDFs is ranked from 1 to 4". Also "of the 10,000 values sampled from a given dataset, roughly 2,500 values in the top 25th percentile are replaced". |
| **Code** | NB3 cell 21 sets `neccs = 1000`. The value 10,000 appears only in cells 17 and 18, which are annotated as being for the PhD defense rather than the manuscript. |
| **Fix** | **Analysis.** Resolved in Stage 1 by moving `neccs` to 10,000, which makes the manuscript text correct as written. No text change needed. |
| **Status** | RESOLVED in Stage 1. `neccs` raised to 10,000; the manuscript text is now correct as written. All pLCA numbers moved; see `HANDOFF_stage-1.md`. |

## 2. Normalization: weighted or unweighted mean

| | |
|---|---|
| **Manuscript** | Two statements that disagree with each other. The glossary says: "'Normalizing' a set of values means dividing each value by its mean so that the mean of the resulting set is 1.0. For example, if the values are [0.1, 0.2, 0.3, 0.4], they are divided by the mean (0.25)". That worked example is an **unweighted** mean and matches the code. But the lognormal offset justification says: "because all datasets are normalized to a **weighted** mean of 1.0, the offset is consistently half the mean across datasets". |
| **Code** | `data / np.mean(data)`, the unweighted mean, in both the synthetic path (`datageneration.random_irregular_dataset`) and the empirical path (NB1 cell 12). Confirmed empirically: `mean_uw` is 1.0 for all 10,000 datasets to floating point precision, while the weighted `mean` has an interquartile range of about 0.978 to 1.022. |
| **Fix** | **Text.** Per Stage 1 amendment A3 the code is correct and stays as it is. The rationale is that the practitioner-facing threshold this study builds is of the form "if reweighting shifts a material by more than X percent of its mean ECC, weighting matters more than the choice of distribution", and a practitioner can compute an unweighted mean from a set of EPDs but cannot compute the market-weighted mean without already knowing the market shares, which is the quantity they lack. Change "weighted mean of 1.0" to "mean of 1.0" in the lognormal offset passage. The glossary is already correct. |
| **Status** | Open. Text edit. |

## 3. CORRECTED. "Mode count" naming

| | |
|---|---|
| **Original claim** | "'Mode Count' is a continuous modality index from `estimate_maxima`, not a count of modes." |
| **Why corrected** | The manuscript **already defines it correctly** in the glossary: "Mode count: Heights (densities) of all local maxima minus heights of all local minima, all divided by the height of the absolute maximum. Local maxima and minima are determined by fitting the data with KDE using Scott's rule-of-thumb to estimate the bandwidth." That is exactly what `customstats.estimate_maxima` computes. There is no false description. |
| **What remains** | The **name** invites a misreading, and the narrative text leans into it: "when the mode count is 1.0, all UQ methods perform similarly well. However, as the mode count increases..." reads as though it counted modes. A reviewer who takes the name at face value will read a non-integer "count" in Figure 4k and 4l and question it. |
| **Fix** | **Text.** Either rename to "modality index" throughout, or add a half-sentence at first use noting that the quantity is continuous and not an integer count. |
| **Status** | Open. Text edit, low effort, worth doing. |

## 4. CORRECTED. Outlier filter on dataset metrics

| | |
|---|---|
| **Original claim** | "The 27.5 percent outlier filter is undescribed." |
| **Why corrected** | The filter **is** described: "because this process still yields datasets with divergent statistical metrics, any dataset with a metric that was a statistical outlier relative to the same metric across the other datasets was removed. In other words, datasets with statistical metrics lying outside IQR +/- 1.5*IQR for each metric were removed." |
| **What remains, and it is substantive** | Two things are absent. (a) **The magnitude.** 4,131 of 15,000 generated datasets, 27.5 percent, are discarded. The manuscript never says how many, and a reader would reasonably assume a few percent. (b) **The IQR is widened.** NB1 cell 20 uses `iqr = np.max([q3-q1, std*1.35])`, not the plain interquartile range the text describes. For several metrics the standard deviation term dominates, so the effective filter is wider than stated. For `kurtosis` the plain IQR is 2.56 while `1.35*std` is 27.71, an order of magnitude difference. |
| **Also unstated** | The filter removes 1,061 datasets on `weight_outliers` and 824 on `n`. Filtering on `weight_outliers` preferentially discards the datasets where the weighting scheme matters most, which is the phenomenon the paper is about. |
| **Fix** | **Text**, unless Stage 2 changes the filter, in which case both. Report the number removed, state the `max(IQR, 1.35*sigma)` definition, and address the selection effect on `weight_outliers`. |
| **Status** | Open. Text edit at minimum. Stage 2 is reviewing the filter itself. |

## 5. Dataset size range

| | |
|---|---|
| **Manuscript** | "the size of the overall dataset, n, was specified by generating random integers between 3 and 1,000 using a logarithmic function". |
| **Code** | `random_logcount(lo=4, hi=1000)`, so the lower bound is 4, not 3. More importantly, the metric outlier filter removes 824 datasets on `n`, capping the effective maximum at **749**. No dataset larger than 749 survives into the analyzed set of 10,000. |
| **Fix** | **Text.** State 4 rather than 3, and report the effective post-filter range of 4 to 749 rather than the pre-filter sampling range. |
| **Status** | Open. Text edit. |

## 6. CORRECTED. Lognormal offset of 0.5

| | |
|---|---|
| **Original claim** | "`logfit_offset = 0.5` is undocumented." |
| **Why corrected** | The manuscript documents it twice, and well: "Because many datasets had values too close to zero for a realistic lognormal distribution fit, the data were offset by 0.5 in the positive direction for distribution fitting", and in the discussion, "this study fits the data after offsetting by 0.5 to achieve stronger fits for datasets with values near zero. This offset is a scale-aware, consistent heuristic." |
| **What remains** | It is undocumented **in the code**. `logfit_offset = 0.5` appears as a bare assignment in NB2 cells 12 and 20 and NB3 cell 15 with no comment. |
| **Fix** | **Code comment**, not text. Note also that the justification quoted above relies on the "weighted mean of 1.0" claim corrected in entry 2; once that wording is fixed the offset argument still holds, since the unweighted mean is exactly 1.0. |
| **Status** | PARTIALLY RESOLVED in Stage 1. The constant is now `LOGFIT_OFFSET` in `src/fitting.py` with a comment explaining what it is and why it exists. The methodological question of whether the threshold should be estimated rather than fixed is Stage 2; see entry 15. |

## 7. The (1-capecc) divisor on cap rank frequencies

| | |
|---|---|
| **Manuscript** | "EC Reduction [Cap ECC / Mat Red] Rank #i Frequency: For each of 10,000 iterations, each of four PDFs is ranked from 1 to 4 based on which dataset resulted in the greatest to least reduction from this intervention. [It] represents the percentage of iterations in which a given dataset is 1st, 2nd, 3rd, and 4th." A plain percentage of iterations. |
| **Code** | NB3 cell 21 computes `df_rank = df_rank.fillna(0)/neccs/(1-capecc)`, dividing by 0.25 as well as by the iteration count, so the reported values are four times a percentage of iterations. The inline comment concedes the problem: "Percentages look off because not all reduction strategies apply in all scenarios". |
| **Fix** | **Settled in Stage 2g: it is a conditional normalization and the manuscript must define it as one.** The denominator is the iterations in which capping SOMETHING helps, not all of them, and the applicability is a reported quantity beside the frequencies rather than a constant folded into them. The matching material-reduction block still has no divisor and needs none, because that strategy applies in every iteration; the manuscript's claim that the two definitions are "equivalent" is what is wrong. |
| **Status** | RESOLVED in Stage 2g. See entry 134 for the corrected definition and what moved. Text edit outstanding. |

## 8. Variance inflation in synthetic data generation

| | |
|---|---|
| **Manuscript** | The generation procedure is described in sequence: sampling `n`, choosing modes with locations, variances and weights, then "Negative numbers were replaced with additional values generated recursively with the process described above... An additional filter in this recursive process was to exclude extreme outliers". |
| **Code** | `datageneration.random_irregular_dataset` contains an undescribed step between those two: `exp = rng.uniform(0.9, 4.0)` followed by raising every value to that power, commented "increase standard deviation of data to align with empirical ECC data". With locations in [5, 20] this maps values as high as 20 to 160,000, and it is the dominant source of right skew in the synthetic data. A reflection step follows, applied with probability 0.25, which is also undescribed. |
| **Fix** | **Text.** Both steps must be described, because between them they determine the skewness and kurtosis distributions that the whole Figure 4 analysis is conditioned on. |
| **Status** | Open. Text addition. Stage 2 is reviewing whether the exponent step should remain. |

## 9. Number of statistical metrics in Figure 4

| | |
|---|---|
| **Manuscript** | "Figure 4 shows the rolling average of W1 distance for each of the six UQ methods as a function of the 18 statistical metrics calculated for each synthetic ECC dataset, namely skewness ()." Panels are referenced individually from Figure 4a through Figure 4r, which is 18 panels. |
| **Code** | The recovered `metrics` definition resolves to **19** metrics, so the figure has 19 panels, a through s. One panel is therefore produced but never referenced in the text. |
| **Also** | The sentence ends "namely skewness ()." with an empty parenthesis, a broken cross-reference that must be repaired regardless. |
| **Fix** | **Text.** Reconcile the count, add the missing panel reference, and repair the broken cross-reference. Confirm against the regenerated figure which metric is unreferenced. |
| **Status** | Open. Text edit. |

## 10. Bandwidth rule, and consistency with the author's own KL2 paper

| | |
|---|---|
| **Manuscript** | Accurate on what the code does: "The bandwidth of each kernel... is determined using Scott's rule-of-thumb, which is a function of the size and standard deviation of a dataset (Scott, 1992). Because KDE is applied to datasets with variable weights, the effective size of the dataset is used". This matches `weighted_bw(..., 'scott')`, which is `1.06 * std * n_eff**-0.2`. |
| **What is missing** | No justification for choosing Scott over Silverman, and no acknowledgement that the author's own KL2 paper (Torres et al., 2026, RC&R 234, 109022) uses Silverman's rule and justifies it explicitly. A reviewer who knows the KL2 paper will ask. Silverman 1986 and Sheather and Jones 1991 are both already in the reference list but neither is discussed on this point. |
| **Separate hazard** | `scipy.stats.gaussian_kde` uses the words "scott" and "silverman" for different formulas than the textbooks: its `'scott'` has no 1.06 factor and its `'silverman'` is approximately `1.06*sigma*n**(-1/5)`, which is this project's `'scott'`. The code sidesteps this correctly. Do not describe the method by pointing at a scipy keyword. |
| **Fix** | **Text.** Add a justification and reconcile with KL2. |
| **Status** | Open. Deferred to Stage 2 as a statistical judgment item. |

## 11. Shapiro-Wilk and Shapiro-Francia presented as one statistic

| | |
|---|---|
| **Manuscript** | "The Shapiro-Wilk test is used to assess goodness-of-fit for each dataset against normal and lognormal distributions." Figure 4e to 4h present the uniform-weighted and variable-weighted versions side by side as though they were the same statistic computed two ways. |
| **Code** | `customstats.shapiro_wilk_weighted` returns `scipy.stats.shapiro`, the true Shapiro-Wilk W, when weights are uniform, but a **Shapiro-Francia** statistic, the squared weighted correlation with normal scores, when weights are non-uniform. So `fit_norm_SW` and `fit_norm_SW_uw` come from two different estimators, and the same holds for the lognormal pair. |
| **Fix** | **Undecided.** Either the text explains that the weighted variant is Shapiro-Francia and argues the two are comparable, or the code uses one estimator consistently for both columns. The function's own docstring notes the two agree closely only for n >= 20; the median dataset size is 62 but the minimum is 4. |
| **Status** | Open. Deferred to Stage 2 as a statistical judgment item. |

## 12. Citation now published

| | |
|---|---|
| **Manuscript** | Two places. Inline: "these functionalities were not explored in this study (Torres et al., in press)". Reference list: "Torres, M. I., Lupton, R., Marsh, E., Srubar, W. V., III, & Allen, S. (in press). Using kernel density estimation and the Dirichlet distribution for uncertainty quantification of building material emissions. Resources, Conservation & Recycling." |
| **Correct citation** | Torres, M. I., Lupton, R., Marsh, E., Srubar, W. V., III, & Allen, S. (2026). Using kernel density estimation and the Dirichlet distribution for uncertainty quantification of building material emissions. *Resources, Conservation and Recycling*, 234, 109022. |
| **Fix** | **Text.** Update both occurrences. No other placeholder or in-press citation was found in the manuscript. |
| **Status** | Open. Text edit. |

---

## New, found during Stage 1

## 13. Infinite kurtosis for five empirical datasets

| | |
|---|---|
| **Manuscript** | Defines kurtosis without qualification: "Kurtosis: How heavy the dataset's tails are, measured as a scaled version of the fourth moment of the distribution." |
| **Code** | `customstats.weighted_kurtosis` applies a bias correction whose denominator contains `(n-3)`. Five of the 138 empirical datasets have exactly n = 3 (`AluminiumSiding`, `Electrical`, `Flooring`, `WallBase`, `WindTurbines`), so their `kurtosis` and `kurtosis_uw` are **infinite** in the shipped `TABLE_EmpiricalECCMetrics.xlsx`. Ten non-finite cells in total. Synthetic datasets are unaffected because their minimum n is 4. |
| **Consequence** | The supplementary empirical metrics table contains `inf`, and the empirical distributions plotted in `CompareUQMethods_SUPP_GeneratedVsEmpiricalMetrics.png` for the two kurtosis panels are computed from a series containing infinities. |
| **Fix** | **Analysis.** Return the biased estimator, or NaN, for n < 4. **Deliberately not fixed in Stage 1**, because the Stage 1 prompt permits exactly two number-moving changes and this would be a third. |
| **Status** | RESOLVED in Stage 1. `weighted_kurtosis` returns NaN for n < 4. Nine cells moved from infinite to NaN; no finite value changed. |

## 14. Synthetic datasets do not cover the empirical size range

| | |
|---|---|
| **Manuscript** | "The purpose of generating synthetic ECC datasets is to increase the generalizability of this study... by increasing the variety of potential datasets beyond that of the 138 empirical ECC datasets". The synthetic sizes are said to align "with the distribution of the empirical ECC datasets". |
| **Code and data** | Empirical dataset sizes run from 3 to **77,548** (`ReadyMix`), with a median of 37. Synthetic sizes run from 4 to **749** after filtering. Six empirical datasets exceed the synthetic maximum, including the two largest and most consequential material categories, `ReadyMix` (77,548 EPDs) and `Asphalt` (8,925). |
| **Consequence** | The claim that the synthetic set spans the empirical range does not hold at the upper end, and it fails precisely where the most data-rich and most carbon-significant material sits. Figure 4m discusses goodness-of-fit as a function of dataset size over a range that stops two orders of magnitude short of the largest real dataset. |
| **Fix** | **Undecided.** Either extend the synthetic size range, or state the limitation explicitly and bound the conclusions to n <= 749. |
| **Status** | Open. Carried to Stage 2. |

## 15. The lognormal threshold, and why the offset exists

| | |
|---|---|
| **Manuscript** | "Because many datasets had values too close to zero for a realistic lognormal distribution fit, the data were offset by 0.5 in the positive direction for distribution fitting." |
| **Investigated in Stage 1** | The pathology is real and was measured. With the threshold forced to zero, a handful of near-zero values drag the log-space mean down and inflate sigma, collapsing the fitted mode toward zero. On `dataset3365` the fitted sigma is 2.54 and the density at the mode is 19.75 against 0.13 at the dataset mean. That is the spike the author remembers. |
| **Scale of the problem** | Worse in the real data than the synthetic. 28.3% of the 138 empirical datasets have a minimum below 1% of their mean, and 19.6% below 0.1%, against 1.8% and 0.20% of the synthetic datasets. |
| **What the offset actually is** | A 3-parameter lognormal with the threshold fixed at -0.5 rather than estimated. Its effect is material: across 1,500 datasets it improves the lognormal fit in 73.9% of cases, median -7.4% W1, and moves the head-to-head against the normal from 35.0% to 43.9%. |
| **Estimating the threshold instead** | Better on average, mean W1 0.12033 against 0.12372 for the fixed offset and 0.15584 for no offset, and better in 76.1% of datasets. Two caveats: as the threshold goes to minus infinity the lognormal converges to a normal, which the author considers a feature rather than a bug, and the MLE still failed on the worst spike case. |
| **Fix** | **Analysis, in Stage 2.** Estimate the threshold. Declare the lognormal as a three-parameter family and state parameter counts plainly, because W1 is an in-sample criterion with no complexity penalty and KDE is more flexible still. Consider a held-out or cross-validated W1 as a robustness check. Fix the cleaning first, entry 16, since that removes much of the pathology at source. |
| **Status** | Open. Stage 2. |

## 16. The low-end cleaning filter never binds

| | |
|---|---|
| **Manuscript** | "Extreme outliers were also discarded, which were defined as outside IQR +/- 3*IQR." |
| **Code** | The bound is computed additively, `Q1 - 3*IQR`, on strictly positive right-skewed data. That bound is **negative in 128 of the 138 empirical datasets (93%)**, so it can never bind. High outliers are removed; low ones never are. |
| **Consequence** | `ReadyMix` retains a value at **3.1e-17 of its mean**, next to a median of 335. A ready-mix concrete EPD reporting that is a data error, not a product. It is also the single value most responsible for the lognormal pathology in entry 15. |
| **Fix** | **Analysis, in Stage 2.** A multiplicative filter, IQR in log space, where the data are roughly symmetric. Tested on ReadyMix it keeps 77,439 of 77,548 values with bounds [90.6, 1240], a physically sensible range, while the additive filter keeps all 77,548. Synthetic generation needs a matching floor so it cannot manufacture values orders of magnitude below the mean either. |
| **Status** | Open. Stage 2. |

## 17. Monte Carlo noise was never quantified

| | |
|---|---|
| **Manuscript** | Reports differences between UQ methods in pLCA results without stating the Monte Carlo uncertainty of those differences. |
| **Measured in Stage 1** | Comparing two runs that differ only in their random draws, at neccs=1000: the noise in `eci_rank_1` is 0.0149 on average, which is 15.5% of that result's standard deviation across datasets, matching the theoretical standard error of 0.0158 for a proportion at n=1000. |
| **Consequence for the central claim** | All 15 pairs of UQ methods differ by 2.2x to 4.0x the noise, so the claim that UQ method choice changes pLCA outcomes does hold. But the weakest pairs sat only 2.2x above noise at the sample size actually used. At neccs=10000 the floor falls to about 0.0047 and those comparisons rise to roughly 7x. |
| **Fix** | **Text.** State the Monte Carlo standard error alongside the reported differences. The numbers above are available in `tests/fixtures/plca/README.md`. |
| **Status** | Open. Text addition, once the Stage 2 regeneration is complete and the figures are final. |

## 18. Scoring grid includes zero

| | |
|---|---|
| **Code** | The W1 scoring grid runs from exactly 0, so the model is truncated to [0, inf) and renormalized, while the pLCA rejection sampling discards draws `<= 0`, giving (0, inf). |
| **Consequence** | Negligible numerically, but the two are not the same set, and the author's position is that zero is not an acceptable ECC. |
| **Fix** | **Code.** Make the scoring grid open at zero so the model scored is exactly the model sampled. |
| **Status** | Open. Stage 2, low priority. |

---

## New, found during Stage 2a

## 19. The two arms used different Dirichlet concentrations

| | |
|---|---|
| **Manuscript** | Describes the market-share weights as drawn from a uniform or flat Dirichlet. |
| **Code** | The synthetic arm used `np.random.dirichlet(np.ones_like(data))`, alpha = 1, which is flat. The empirical arm used `np.random.dirichlet(np.ones_like(data)*5)`, alpha = 5, which is not. The two arms of the study therefore received systematically different weight concentrations, on the exact dimension the paper is about. |
| **Consequence** | Large, and on the headline quantity. Redrawing the empirical weights at alpha = 1 more than doubles the mean uniform-to-variable Wasserstein-1 distance across the 138 datasets, from **0.0594 to 0.1329**, a shift of 1.33 standard deviations of the alpha = 5 distribution. Kurtosis moves 0.98 sd, weight_outliers 0.58 sd, skewness 0.63 sd. Every published empirical-versus-synthetic comparison of the weighting effect was made between arms that were not comparable. |
| **A wording trap** | alpha is the Dirichlet CONCENTRATION parameter, and a SMALLER alpha gives MORE dispersed market shares. So alpha = 1 makes the weights less equal than alpha = 5 did and the measured weighting effect goes UP. Do not write that alpha = 1 is "less concentrated" without saying which sense is meant. The lower-bound argument still holds, but for a different reason: at n = 100 a flat Dirichlet gives an expected largest share of 5.2 percent, while Marsh, Hattam and Allen (2025) report Rest-of-World BOF steel at 63.75 percent of global production, so alpha = 1 still understates real market concentration considerably. |
| **Fix** | **Code, done in Stage 2a.** alpha = 1 in both arms. The text is already correct; the numbers move. |
| **Status** | Resolved in code. The manuscript's empirical weighting-effect numbers must be replaced. |

## 20. The undocumented 27.5 percent selection step

| | |
|---|---|
| **Manuscript** | Describes generating 10,000 synthetic datasets. It does not describe any selection. |
| **Code** | 15,000 were generated, then every dataset that was a marginal outlier on any of 20 metrics was discarded, using Q1 - 1.5*IQR and Q3 + 1.5*IQR with the IQR widened to `max(q3-q1, std*1.35)`, and the first 10,000 survivors kept. **4,131 datasets, 27.5 percent, were discarded.** |
| **Consequence** | It removed exactly the cases the paper exists to study. `weight_outliers` flagged more datasets than any other metric (1,061), and the removed set averaged 0.0755 against 0.0146 for the kept set, a difference of 0.83 pooled standard deviations. The removed datasets were also less normal (fit_norm_SW -0.89 sd), more variable (coeffvar +0.48 sd), larger (n +0.52 sd), more multimodal (+0.46 sd) and had a larger uniform-to-variable W1 (+0.54 sd). The filter flagged 824 datasets on `n` alone, every one with n >= 750, which is what capped the analyzed maximum at 749 against a stated 1,000. It also flagged 14 datasets on `mean_uw`, a column that is 1.0 by construction and ranges only from 0.99999999999999911 to 1.0000000000000009. |
| **Fix** | **Code, done in Stage 2a.** Replaced by a validity-only filter that rejects a dataset solely because it cannot be analyzed. Empirical plausibility is now a reported coverage statistic, not an enforced criterion. |
| **Status** | Resolved in code. The manuscript must either describe what the old corpus was or, preferably, describe the new one. |

## 21. "Mode Count" is not a count

| | |
|---|---|
| **Manuscript** | Reports a metric called "Mode Count". |
| **Code** | `estimate_maxima` returns `(sum of KDE local maxima heights - sum of local minima heights) / max height`, a continuous modality index. Across the 138 empirical datasets it spans only **1.000 to 1.159**, so read as a count it is constant at 1 for every empirical dataset. |
| **Consequence** | The metric could not see multimodality in the empirical data at all. Silverman's critical-bandwidth test finds **27 of the 138 empirical datasets multimodal**: 22 bimodal, 4 trimodal and 1 with four modes. Rounded, the old index agrees with the Silverman count in 80.4 percent of empirical and 53.2 percent of synthetic datasets; Spearman correlation between the two is 0.673. |
| **Fix** | **Code and text, done in Stage 2a.** The column is renamed `modality_index`, which is what it measures, and `crit_bw_1` is added alongside it. Stage 2f decides which survives into the final metric set. |
| **Status** | Resolved in code. The manuscript's "Mode Count" label and any claim resting on it must change. |

## 22. The power transform and the reflection were undescribed tuning steps

| | |
|---|---|
| **Manuscript** | Describes the synthetic datasets as mixtures of named component distributions. |
| **Code** | After drawing the mixture, every value was raised to a power drawn from U(0.9, 4.0), with an inline comment saying the purpose was to align with empirical ECC data, and 25 percent of datasets were then reflected as `np.max(data) - data + np.min(data)`. Neither step is in the manuscript. The reflection used realized order statistics, so the distribution a synthetic dataset came from could not be written down. With locations up to 20 and exponents up to 4 the power transform mapped values as high as 160,000. |
| **Fix** | **Code, done in Stage 2a.** Both removed. Skewness is now a component moment target, and left skew comes from reflected one-sided families, which is a population property. |
| **Status** | Resolved in code. The manuscript's description of data generation must be rewritten against src/genconfig.py and Table 1. |

## 23. The empirical datasets are not deduplicated at product level

| | |
|---|---|
| **Manuscript** | Treats each EC3 category as a set of independent product EPDs. |
| **Measured in Stage 2a** | At EPD level there is nothing to deduplicate: 0 of 206,668 records share an `open_xpd_uuid` within their category, and no EPD appears in two of the 138 categories. But **55.00 percent of records share a (manufacturer, GWP per kg) pair with another record in the same category**, across 105 of 138 categories, and the top manufacturer holds a median 18.4 percent of a category, up to 73.0 percent. |
| **Consequence** | The uniform-weighted empirical distribution the paper scores against is already implicitly weighted, by how many EPDs each manufacturer published. That bears directly on the paper's thesis about weighting: the "unweighted" baseline is not weight-free. |
| **Caveat** | Measured on a 2026-08 EC3 pull rather than the 2026-03 pull that produced the 138 datasets. EC3's contents change over time, so the shares will differ somewhat on any other pull; the finding that manufacturers are heavily duplicated is structural and will not. |
| **Fix** | **Text at minimum.** State that the uniform-weighted baseline carries publication-frequency weighting. Whether to deduplicate is a methodological choice for a later stage. |
| **Status** | Open. |

## 24. No industry-average EPDs are mixed in

| | |
|---|---|
| **Question** | Whether industry-average and product-specific EPDs are mixed within a category. |
| **Measured in Stage 2a** | They are not. All 206,668 records in the 138 categories are Product EPDs. Within those, 98.5 percent are product-specific and 82.9 percent plant-specific, so 16.7 percent are manufacturer-level rather than plant-level, but no industry-average declaration appears. |
| **Fix** | **Text, optional.** This can be stated positively as a data-quality property of the extraction. |
| **Status** | Resolved, no change needed beyond an optional sentence. |

## 25. The cleaning rule moves the empirical metric ranges a lot for what it removes

| | |
|---|---|
| **Manuscript** | "Extreme outliers were also discarded, which were defined as outside IQR +/- 3*IQR." |
| **Measured in Stage 2a** | The rule removes only 1.61 percent of records but moves `fit_norm_SW` by 1.55, `entropy` by 1.42, `modality_index` by 1.00 and `weight_outliers` by 0.77 standard deviations of the uncleaned metric. A multiplicative (log-space) 3*IQR rule removes less (1.20 percent) and moves entropy and fit_norm_SW less, at the cost of moving fit_lognorm_SW more. |
| **Consequence** | The cleaning rule is not a minor tidy-up. It substantially determines the empirical metric ranges that anchor the whole study, so it has to be stated precisely and its sensitivity reported. |
| **Fix** | **Text.** Report the sensitivity. See `outputs/tables/audits/TABLE_2a_EmpiricalCleaningSensitivity.csv`. |
| **Status** | Open. Text. |

## 26. Pull fresh EC3 data

| | |
|---|---|
| **Code** | Notebook 1's empirical branch reads `'../../EPDsFromEC3/EPD_AllOfEC3'`, a path that does not exist. The working copy of that project is at `../EPDsFromEC3` and its EPD store has moved to a different layout. That path is dead code and should be corrected or removed. |
| **What the analysis currently uses** | `dct_realeccs_trimmed.json`, a stored 2026-03 pull, already trimmed additively at the high end before it was written. Because it is stored post-cleaning, Stage 2a could only apply a multiplicative bound to the LOW end; applying one to the high end would trim the same tail twice. |
| **Fix** | **Take a fresh EC3 pull and use it.** The API key and a documented procedure are in `../EPDsFromEC3`, and `PULLING_EPDS.md` records three ways a paginated pull fails while reporting success. With raw values in hand, apply the symmetric log-space cleaning rule, which the sensitivity analysis prefers: the choice of rule moves `fit_norm_SW` by about 1.5 standard deviations and `entropy` by about 1.4, so it is not cosmetic. Record the pull date and the store manifest alongside the result. |
| **Note** | There is no reason to try to reconstruct the 2026-03 snapshot. The analysis is being redone and decision 5 already accepts that the manuscript's numbers are invalidated, so current data is what is wanted. (It could be approximated by filtering on `date_of_issue` and `date_validity_ends`, but nothing needs it.) |
| **Consequence** | Re-pulling moves every empirical number, including the headline uniform-versus-variable W1. That is expected and acceptable; it needs to happen once, deliberately, and be recorded. |
| **Status** | Open. Recommended action, needs the author's go-ahead because it moves every empirical number. |

## 27. Component separation was far outside the empirical range

| | |
|---|---|
| **Manuscript** | Describes multimodal synthetic datasets without characterizing how separated the modes are. |
| **Measured in Stage 2a** | The old generator placed components a mean of **6.46 pooled standard deviations apart** (median 6.01, up to 20.95), producing well-separated clusters. Fitting a BIC-selected Gaussian mixture to every dataset and computing the Maitra-Melnykov pairwise overlap on the same footing gives an empirical median overlap of 0.0218 against a shipped synthetic median of 0.0037: the synthetic datasets had about **six times less mode overlap** than the empirical ones at the median. |
| **Fix** | **Code, done in Stage 2a.** Overlap is now a generation parameter, drawn log-uniformly on [1e-4, 0.75], which covers the empirical maximum of 0.6719 with margin. |
| **Status** | Resolved in code. The manuscript should report the overlap distribution, and Table 1 gives it. |

---

## New, found during Stage 2a-2

## 28. The empirical arm is 136 categories, not 138

| | |
|---|---|
| **Manuscript** | States 138 empirical ECC datasets throughout, and the count appears in figure captions and in the abstract's framing of the empirical arm. **The current count is 149; see CURRENT CANONICAL NUMBERS above.** |
| **Code** | The 2026-08 extract, cleaned symmetrically and filtered to categories retaining at least three values, yields **136**. `Siding` retains 1 value and `SinglePlyOther` retains 2. Both losses are expiry: `Siding` holds 29 records in the store slice of which 1 is still valid at the pull date, and only 14 of the 29 carry a parseable declared unit. |
| **Fix** | **Text.** Replace 138 with 136 everywhere, and state the inclusion threshold (at least three values after cleaning) so the number is derivable rather than asserted. |
| **SUPERSEDED by Stage 2a-3** | The arm is now **149 datasets**. The 136 categories are resolved into specifiable products: 15 EC3 residual bins dropped, concrete split by specified compressive strength, insulation by material type. `outputs/tables/TABLE_EmpiricalCategorySplit.csv` is what makes 149 derivable. |
| **Status** | Open. Text edit, and it appears in many places. The number to write is 149 datasets, drawn from 138 EC3 categories of which 136 survived cleaning and 121 survived the residual-bin rule before splitting. |

## 29. The empirical characteristics were substantially an artifact of the cleaning rule

| | |
|---|---|
| **Manuscript** | Reports the empirical statistical characteristics as properties of the EC3 data, and the synthetic corpus is justified by covering them. |
| **Measured in Stage 2a-2** | The 2026-03 file was trimmed additively at the high end before it was stored, which removes right tail. Re-extracting raw values and applying the multiplicative rule symmetrically moves the empirical characteristics a long way: |
| | median coefficient of variation **0.600 to 0.782**; log10 standard deviation of it 0.2913 to 0.3752; maximum 2.40 to 13.40 |
| | median skewness **1.055 to 2.060**; maximum 4.618 to 20.65 |
| | median excess kurtosis **1.160 to 5.758**; maximum 62.7 to 475.2 |
| | median dataset size 37 to 53 |
| | share of datasets Silverman's test calls unimodal **81.9 percent to 49.3 percent** |
| **Consequence** | Every statement in the manuscript about what real ECC datasets look like was measured on data whose right tail had been cut. The multimodality figure is the one that matters most: the paper's central comparison is between a KDE, which can represent a second mode, and parametric fits, which cannot, and the empirical prevalence of multimodality roughly doubled. |
| **Caveat, and it is not small** | Part of the movement is the newer pull rather than the cleaning rule; the two are separated in `reports/HANDOFF_stage-2a2.md` and in `outputs/tables/audits/TABLE_2a2_FourWayComparison.csv`. |
| **Fix** | **Text.** Every empirical characteristic number is replaced. State the cleaning rule precisely, in log space and symmetric, and report its sensitivity from `TABLE_2a2_CleaningSensitivity.csv`. |
| **Status** | Open. Supersedes the numbers in entries 25 and 27. |

## 30. The empirical data is an archived extract, and should be cited as one

| | |
|---|---|
| **Manuscript** | Describes the empirical data as extracted from the EC3 API. |
| **Code** | The empirical arm reads `data/raw/ec3_raw_ecc_<pull date>.csv.gz`, a frozen extract of the consolidated EPD store, pulled 2026-08-13/14 and checksummed in `data/INPUTS.sha256` with its query and pull dates. |
| **Consequence** | For a reader this is an improvement: EC3's contents change as declarations are issued and expire, so a live query is not reproducible while an archived extract is. |
| **Fix** | **Text.** Cite the archived file and its pull date rather than implying the reader can re-run the query and obtain the same data. |
| **Status** | Open. Text edit. |

## 31. Some EC3 categories span several orders of magnitude and are not one population

| | |
|---|---|
| **Measured in Stage 2a-2** | With the right tail no longer cut, `PowerCabling` runs from 1.7e-05 to 242 times its own mean over 400 values, `Aggregates` from 3.2e-06 to 265 over 385, and `Insulation` from 3.0e-04 to 99 over 666. Their coefficients of variation are 13.4, 10.3 and 7.8 against a median of 0.78 across the 136. |
| **Consequence** | These are not outliers the cleaning rule failed to catch; the log-space interquartile range of such a category is genuinely enormous, so a 3 x IQR bound is very permissive on it. They are EC3 categories holding products that are not comparable, cable of different gauges being the clearest case. They dominate the upper tail of every characteristic and therefore stretch the envelope the synthetic corpus is asked to cover. |
| **Fix** | **RESOLVED in Stage 2a-3, decision 46.** The categories are resolved into specifiable products by three rules that read only metadata, never the ECC values: EC3 residual bins are dropped, concrete is split by specified 28-day compressive strength, insulation by material type from the product name. Arm 136 to 149. |
| **The framing the text must use** | **This is not a dispersion fix and must not be written as one.** Splitting `ReadyMix` by strength moves its coefficient of variation only from 0.29 to 0.27 and is still right, because 4000 psi and 5000 psi concrete are different products and strength is the primary characteristic a structural engineer specifies concrete by. A dataset stands for one material choice in a pLCA, so the test is whether a category is something a specifier could name. |
| **What the text must say** | The three rules; that EC3 records no subcategory on any of these EPDs, `category_key` equalling the queried category for all 123,060, so the tree's parent/child relation rather than a per-record field is what identifies a residual bin; that "type not stated" is a real dataset of insulation EPDs whose name does not state a material, 131 of 335 board records; that thickness was tested and rejected because it parses for only 96 of 335; and the four parent categories KEPT because no child of theirs is in the arm, whose heterogeneity is a limitation. |
| **Effect on the envelope** | Median coefficient of variation 0.757 to 0.667, median excess kurtosis 5.49 to 3.72, median skewness 1.73 to 1.55, Silverman unimodal share 49.3 to 55.7 percent, maximum dataset size 86,770 to 31,025. The visible-mode distribution moves by 0.0045. |
| **Status** | Resolved in code. Text owes the three rules, the framing above, and the table. |


---

## CURRENT CANONICAL NUMBERS, as of the Stage 2c review closing, 2026-09-17

**Read this before working from any entry below.** Entries are appended and never
rewritten, so an older one may quote a figure that a later stage has moved. This
block is the single place to check what a number currently is. Anything here
beats anything below it.

**Every entry from 1 to 27 was written against the 2026-03 empirical data and the
pre-regeneration corpus. Treat their NUMBERS as historical and their ARGUMENTS as
live.** Where such an entry says "the 138 empirical datasets", the count is now
149 and the values differ; the point the entry is making usually still stands.

### The empirical arm

| | |
|---|---|
| **datasets** | **147** |
| drawn from | 138 EC3 categories queried; 136 retained at least 3 values after cleaning; 121 survived the residual-bin rule; splitting concrete and insulation brings it to 149; dropping `Chairs` and `Grouting` as not one product population brings it to 147 (decision 61) |
| source | `data/raw/ec3_raw_ecc_2026-08-14.csv.gz`, a frozen archived extract, pulled 2026-08-13/14, checksummed in `data/INPUTS.sha256` |
| ECC values after cleaning | **116,766** (117,090 before the plausibility ceiling; 117,079 before the category rules of decision 61; 116,768 before the declared-unit consistency check of decision 63) |
| cleaning | multiplicative 3 x IQR in LOG space, both ends, after an external plausibility ceiling of 100 kgCO2e/kg on mass-declared records (entry 35). **The log-space rule works on a coherent category and fails on a contaminated one**, because its width is set by the spread of the contamination: on `ReadyMix [4000-4999 psi]` its upper bound is 3x the median and it trims 42 records; on `Aggregates` it was 41,238,610x the median and trimmed nothing (entry 51) |
| weighting | flat Dirichlet, alpha = 1, keyed by dataset name |
| normalization | each dataset divided by its own UNWEIGHTED mean |

Per-dataset characteristics, median / min / max:

| characteristic | median | min | max |
|---|---|---|---|
| coefficient of variation | 0.658 | 0.006 | **6.929** (`Aggregates`; was 13.404 for `PowerCabling` before decision 63 removed two mislabeled records) |
| skewness | 1.539 | -2.408 | **21.000** |
| excess kurtosis | 3.702 | -3.627 | **525.17** |
| entropy | 2.967 | 0.233 | 4.883 |
| weight of outliers | 0.041 | 0.000 | 0.352 |
| `fit_norm_SW` | 0.850 | 0.039 | 0.997 |
| `fit_lognorm_SW` | 0.947 | 0.759 | 1.000 |
| `w_v_uw_wasserstein` | 0.094 | 0.001 | 0.730 |
| `crit_bw_1` | 0.710 | 0.215 | 4.176 |
| modality index | 1.004 | 1.000 | 1.074 |
| dataset size n | 50 | 3 | 31,025 |

Modality, and **quote `nboot` whenever quoting the Silverman share**: 55.7 percent
unimodal by Silverman's critical-bandwidth test at nboot = 100; by VISIBLE modes,
95.3 percent have one, 4.0 percent two, 0.7 percent three or more.

### The synthetic corpus

**`corpus_2026-09-15b`, seed 42, 10,000 datasets**, plus a 50-dataset probe set
held outside every aggregate. Stratified 2,500 per stratum over n = 3-9, 10-99,
100-999 and 1000-9999. Decision 55 regenerated it and decision 58 recomputed its
characteristics without redrawing anything; **the "9,999, not 10,000" of
`corpus_2026-09-14d` no longer applies and the pLCA is 2,500 groups again.**

Match against the empirical arm at the time of tuning: mean standardized W1
across the ten characteristics **0.2326**; visible-mode total variation
**0.0128**. Against the 147-dataset arm the weighted objective reads 0.2425, and
**decision 63 explains why that is a standardization artifact and not a reason to
retune.**

**The parent of every dataset is recoverable exactly**, by replaying the
generator: `corpus.rebuild_parents`, cached as `parents_spec.json.gz`. Decision
64. It is not stored in `parents.json.gz`, which holds only how each parent was
asked for.

### What Stage 2b changed, and the numbers it adds

**READ THE STAGE 2c BLOCK BELOW FIRST.** Three rows in this table are twice
superseded -- by the bandwidth switch of decision 54 and by the arm and corpus
changes of decisions 55 and 61 -- and every W1 in it is against the CIRCULAR
in-sample target that Stage 2c replaced.

| | |
|---|---|
| **empirical values** | 117,090 to **117,079**, and six datasets move. Entry 35 |
| **pLCAs** | **2,499, not 2,500**, over groups of four covering 9,996 of 9,999 datasets. Entry 39 |
| **W1, empirical, mean rank over the six methods** | `Lognormal, Variable` **2.09**, `KDE, Variable` 2.74, `Lognormal, Uniform` 3.34, `KDE, Uniform` 3.52, `Normal, Variable` 4.21, `Normal, Uniform` 5.11. **SUPERSEDED as a headline by Stage 2c: these are IN-SAMPLE numbers against a circular target, and they predate the guarded Silverman bandwidth. See the Stage 2c block below.** |
| **W1, synthetic, mean rank** | `KDE, Variable` **2.12**, `Lognormal, Variable` 2.19, `Normal, Variable` 3.64, `KDE, Uniform` 3.93, `Lognormal, Uniform` 4.17, `Normal, Uniform` 4.95. **SUPERSEDED, same two reasons.** In sample on the current corpus and bandwidth: `KDE, Variable` 1.45, `Lognormal, Variable` 2.48, `KDE, Uniform` 3.81, `Normal, Variable` 3.86, `Lognormal, Uniform` 4.36, `Normal, Uniform` 5.05 |
| **W1, empirical, mean** | `Lognormal, Variable` 0.178, `Lognormal, Uniform` 0.211, `KDE, Variable` 0.251, `KDE, Uniform` 0.286, `Normal, Variable` 0.436, `Normal, Uniform` 0.490. **SUPERSEDED TWICE: these are SCOTT-bandwidth numbers on the 149-dataset arm.** Under the guarded Silverman rule of decision 54 and the 147-dataset arm of decision 61 the in-sample means are `KDE, Variable` **0.1397**, `Lognormal, Variable` 0.1672, `KDE, Uniform` 0.1748, `Lognormal, Uniform` 0.1975, `Normal, Variable` 0.3586, `Normal, Uniform` 0.3957. And in-sample is the circular target; see the Stage 2c block |
| **W1, synthetic, mean** | `Lognormal, Variable` 0.0988, `KDE, Variable` 0.1017, `Lognormal, Uniform` 0.1676, `KDE, Uniform` 0.1711, `Normal, Variable` 0.1724, `Normal, Uniform` 0.2187. **SUPERSEDED: Scott bandwidth, and `corpus_2026-09-14d`.** Under the guarded Silverman rule on `corpus_2026-09-15b`: `KDE, Variable` **0.0776**, `Lognormal, Variable` 0.0986, `KDE, Uniform` 0.1612, `Lognormal, Uniform` 0.1673, `Normal, Variable` 0.1723, `Normal, Uniform` 0.2188 |
| **the lognormal** | 3-parameter, threshold by profile likelihood, guard at 0.25 weighted standard deviations below min(x); the +0.5 offset is retired. Entries 37, 40, 43 |
| **the support** | (0, inf) open at zero, every method truncated and renormalized, sampling by inverse CDF. Decision 13, confirmed |

**The empirical W1 values above are NOT comparable to anything in the manuscript**,
which reports the same quantity in raw category units. Entry 41.

### What Stage 2c changed, and these are the numbers to quote

**The in-sample scores above are against a circular target and are kept only as
the "before".** Every headline below is on a target the model has not seen.

| | |
|---|---|
| **synthetic, W1 against the parent, mean** | `KDE, Uniform` **0.1228**, `Lognormal, Uniform` 0.1306, `KDE, Variable` 0.1639, `Lognormal, Variable` 0.1699, `Normal, Uniform` 0.2103, `Normal, Variable` 0.2368 |
| **synthetic, mean rank against the parent** | `KDE, Uniform` **2.03**, `KDE, Variable` 2.96, `Lognormal, Uniform` 3.00, `Lognormal, Variable` 3.72, `Normal, Uniform` 4.48, `Normal, Variable` 4.81 |
| **empirical, cross-validated W1, mean** | `Lognormal, Uniform` **0.2908**, `Lognormal, Variable` 0.3088, `KDE, Uniform` 0.3214, `KDE, Variable` 0.3411, `Normal, Variable` 0.4739, `Normal, Uniform` 0.4810. **127 of the 147 datasets reach n = 10.** The third decimal moves with the random splits; see entry 54 |
| **the two arms disagree about the FAMILY, and the criterion plus the size mix explain the sign** | Entry 54. What both agree on: the KDE loses at n = 10-99 and wins at n >= 1000 |
| **weighting, on the common market parent** | A coin flip overall -- variable wins 51.1 pct (normal), 54.6 (lognormal), 54.2 (KDE) -- and a size effect underneath: for the KDE, 39.0 pct at n = 3-9 and 75.8 pct at n >= 1000. Entry 55 |
| **the definitional share of a uniform method's in-sample score** | **61.9 pct** on the empirical arm, 85.2 pct on the synthetic. Entry 55 |
| **regret, synthetic against the parent, mean** | `KDE, Uniform` **0.0290**, `Lognormal, Uniform` 0.0368, then the variable pair, then the normals. At p95 the lognormal is tighter, 0.1281 against 0.1506. Empirical, cross-validated: `Lognormal, Uniform` **0.0244** and the tightest p95. Entry 56 |
| **empirical size shares** | 20 / 78 / 38 / 8 over 147, plus three above n = 9,999. Entry 62 |
| **the KDE's advantage is a SHAPE advantage, not a modality one** | Entry 57 |
| **the three-parameter lognormal is indistinguishable from gamma on real data** | Entry 59 |

**Every aggregate is reported equally allocated AND reweighted to the empirical
size mix**, and the reweighting flips the corpus's family verdict on the mean
while leaving it on the rank. Entry 62.

### What the Stage 2c REVIEW changed, 2026-09-17, and these supersede the block above

Three settings moved and all three numbers below come from the notebook tables,
not from an audit script.

| | |
|---|---|
| **the bandwidth guard** | `SILVERMAN_MIN_NEFF` **30 to 20**. Decision 80, superseding 75 |
| **the scoring grid** | **20,000 points, trapezoid quadrature**, from 1,000 atoms. Decision 81. The atom route never converges |
| **W1, empirical, in-sample mean** | `KDE, Variable` **0.1319** (was 0.1471), `Lognormal, Variable` 0.1683, `KDE, Uniform` 0.1727 (was 0.1806), `Lognormal, Uniform` 0.1985, `Normal, Variable` 0.3599, `Normal, Uniform` 0.3970. **Only the KDE moves materially**, -10.4 and -4.4 percent, because the coarse grid was inflating it |
| **W1, synthetic, against the parent, mean** | `KDE, Uniform` **0.1220**, `Lognormal, Uniform` 0.1306, `KDE, Variable` 0.1631, `Lognormal, Variable` 0.1699, `Normal, Uniform` 0.2103, `Normal, Variable` 0.2368 |
| **W1, empirical, cross-validated, mean** | `Lognormal, Uniform` **0.2950**, `Lognormal, Variable` 0.3129, `KDE, Uniform` 0.3198, `KDE, Variable` 0.3437, `Normal, Variable` 0.4789, `Normal, Uniform` 0.4851 |
| **the paired out-of-sample gaps** | synthetic parent, lognormal minus KDE: **+0.0086** uniform, **+0.0068** variable, both distinguishable. Empirical cross-validated: **-0.0247** and **-0.0309**, both distinguishable |
| **the headline by material** | structural categories at n >= 100, 23 datasets and 83 percent of the arm's values: the KDE closest on **69.6 percent** under uniform weighting. Entry 73 |
| **visible modes, empirical** | **68.5 percent** with one mode at the bandwidth the study fits, 94.6 at scipy's default. Entry 70 |
| **the pLCA** | only the two KDE methods move, 12.3 percent of rows, mean `eci_rank_1` change 0.0016. The lognormal and normal rows are bit-identical |

### What Stage 2a-3 changed, entry by entry

| entry | status |
|---|---|
| 28, dataset count | **149**, superseding 136 and 138 |
| 31, categories that are not one product | RESOLVED by three metadata rules; see the entry for the framing the text must use |
| 32, Dirichlet weight sensitivity | NEW, open, owner 2h |
| 33, EC3 records no subcategory | NEW, one sentence for the data section |
| 34, coverage claim is false | NEW, **decided: option A**, needs a text edit and a rebuilt figure |
| 29, empirical characteristics | its 2026-08 numbers are superseded by the table above |
| 25, 27, cleaning and overlap | arguments live, numbers historical |

### The three things the manuscript owes that are NOT yet written anywhere else

1. **The dataset count is 149 and it appears throughout**, including figure
   captions and the abstract's framing.
2. **The coverage claim must be restated** and
   `outputs/figures/CompareUQMethods_FIG_MetricCoverage.png` rebuilt. Entry 34.
3. **The category resolution must be described**, and described as a question of
   what a dataset MEANS rather than as a dispersion fix. Entry 31.

---

## New, found during Stage 2a-3

## 32. A single Dirichlet weight realization moves the per-dataset metrics a long way

| | |
|---|---|
| **Manuscript** | Reports per-dataset weighted characteristics, `w_v_uw_wasserstein` above all, as properties of the dataset. |
| **Measured in Stage 2a-3** | They are properties of the dataset AND of the one Dirichlet draw that produced its weights. Redrawing the weights of the same 136 datasets from the same distribution, changing nothing else, moves `w_v_uw_wasserstein` by up to **1.02** in absolute terms, `coeffvar` by up to 4.09, `skewness` by up to 9.88 and excess kurtosis by up to 366. Every UNWEIGHTED column is bit-identical across the two realizations, which is the proof that the values did not change and only the weights did. |
| **How it was found** | Splitting six categories shifted the position of every later category in the draw order, and the weighted metrics of 130 untouched datasets moved. The weights are now keyed by dataset name rather than by iteration order, so a dataset's weights are a property of that dataset; after the change, splitting moves the 130 shared datasets by exactly zero. |
| **Consequence** | Any per-dataset weighted number in the paper is one draw from a distribution whose spread has never been reported. The AGGREGATE distribution across the arm is far more stable than any single dataset's value, and that is what the study actually rests on, but the distinction is not currently made in the text. |
| **Fix** | **Text, and an analysis decision that belongs to Stage 2h**, which already owns "multiple weight realizations". Report the arm-level characteristic distributions rather than per-dataset weighted values, or report per-dataset values with an interval over weight realizations. |
| **Status** | Open. Owner: 2h for the analysis, text once 2h reports. |

## 33. EC3 carries no subcategory for these records, and that is worth one sentence

| | |
|---|---|
| **Measured in Stage 2a-3** | Across all 123,060 usable records in the 138 queried categories, `category_key` equals the queried category and the finer `category` field is empty throughout. There is no EC3 subcategory to group products by. Exhaustive search of all 106 store columns for the seven screened categories found no product-type field populated for 90 percent or more of records with more than one level; the only such fields are declarer attributes (program operator, PCR, jurisdiction, plant specificity, uncertainty factor). |
| **Consequence** | A reader will reasonably ask why heterogeneous categories were not split on product type. The answer is that EC3 does not record one for these products, not that it was not tried. |
| **Fix** | **Text, one or two sentences**, in the data section, and it strengthens rather than weakens the account: it is why the declared unit is the axis used, and why `Insulation` is left whole. |
| **Status** | Open. Text. |

## 34. The coverage claim is false at the top of the coefficient of variation

| | |
|---|---|
| **Manuscript** | Claims the synthetic corpus covers the region of characteristic space the empirical datasets occupy and extends beyond it on every side, which is what licenses generalizing the study's conclusions past the sampled categories. CLAUDE.md decision 29 records 100 percent coverage on all nine statistical characteristics, approved from `CompareUQMethods_FIG_MetricCoverage.png`. |
| **Measured in Stage 2a-3** | That figure was measured on the Stage 2a empirical arm, whose maximum coefficient of variation was 2.40. Stage 2a-2 rebuilt the arm from raw values and the maximum became 13.40; nothing re-checked coverage. **Empirical datasets have a coefficient of variation the corpus never reaches**, remeasured on the 149-dataset arm: the arm maximum is 14.34 (`PowerCabling`), `Insulation` 6.05, `ConcreteAdmixtures [1 kg]` 3.56, `Grouting [1 kg]` 3.24, `DampproofingAndWaterproofing` 2.50, `WallFinishes` 2.20, against a synthetic maximum of 2.18. Two more are uncovered on `fit_norm_SW`, and `ReadyMix` on `n` by the deliberate 9,999 ceiling of decision 19. |
| **Cause** | Not the draw range, which reaches 16. The coefficient of variation is a POPULATION target while the characteristic measured is the SAMPLE value, which runs low on a right-skewed distribution; only 41.7 percent of targets are met. |
| **Fix** | See the three options below. |
| **Status** | **DECIDED 2026-09-14: option A** (CLAUDE.md decision 48). B and C are declined. What remains is a TEXT edit: restate the coverage claim as measured, name the exceptions, and rebuild `CompareUQMethods_FIG_MetricCoverage.png`. Decision 29 is corrected in place. |

**RESTATED 2026-09-16, AND EVERY NUMBER ABOVE IS SUPERSEDED. DO NOT QUOTE THE
ROWS ABOVE THIS LINE.** Decision 48's premise was partly wrong, through no fault
of its reasoning: the empirical maximum coefficient of variation it was
measuring against, 14.34, was inflated by two mislabeled records. Decisions 61
and 63 removed them and dropped two categories that were not one product
population. **The arm is 147 datasets and 116,766 values.** Current, measured
from the rebuilt `outputs/tables/TABLE_MetricCoverage.csv`:

| | then (149 datasets) | now (147 datasets) |
|---|---|---|
| arm maximum coefficient of variation | 14.34, attributed to `PowerCabling` | **6.93, `Aggregates`** |
| synthetic maximum | 2.18 | 2.58 |
| uncovered dataset-metric pairs | 11 of 1,490 | **5 of 1,470** |
| uncovered on dispersion | 5 datasets | **1 dataset** |
| uncovered on `n` | 3 | 3, by the deliberate 9,999 ceiling of decision 19 |
| uncovered on `fit_norm_SW` | 2 | 1 |

**What the text should now say.** The corpus covers the empirical characteristic
space with margin except on three counts: one dataset of 147 sits above the
synthetic range on dispersion, one on `fit_norm_SW`, and three exceed 9,999
values per dataset, which the probe set covers by design. That is a materially
stronger claim than the one option A was chosen to defend, and the gap that
remains is about half what decision 48 measured.

**Option B is still declined and the reasoning still holds**, but its arithmetic
changes: the target to reach is 6.93 rather than 14.34, and the eight swept
candidates reached 2.15. Whether a generator redesign could now close a halved
gap has NOT been measured. It remains out of scope unless the author reopens it.

**`CompareUQMethods_FIG_MetricCoverage.png` has been rebuilt** and shows the
current state; it is the figure to use.

**THE DECISION, stated as three options.** Earlier versions of this entry said
"undecided and it needs one" without saying what was on offer, which is a defect
in the document rather than a hard question.

| option | what it means | cost |
|---|---|---|
| **A. Change the text** (recommended) | State coverage as measured and name the exceptions. The claim becomes: the corpus covers the empirical characteristic space with margin except at the extreme upper tail of dispersion, where 5 of 149 datasets sit beyond it, and above 9,999 values per dataset, which the probe set covers by design | nothing; no regeneration |
| B. Widen the generator and regenerate | **MEASURED AND NOT AVAILABLE AS A PARAMETER CHANGE.** Eight candidates were swept in `audits/dispersion_reach.py`: raising the target center by 0.4, the spread by 1.8x, the upper truncation to 60, and relaxing the quartile-ratio floor `min_q1_over_iqr` from 0.5 through 0.1, 0.05 to 0.01. **The achieved sample coefficient of variation moves from 1.65 to at most 2.15**, against an empirical maximum of 14.34, and NONE of the eight puts a single dataset above 3 | reaching the empirical tail needs heavier-tailed parents or a different truncation rule, which is a generator REDESIGN, not a retune and not one regeneration |
| C. Exclude the uncovered categories | Drops `Aggregates`, `Chairs`, `Elevators`, `Grouting`, `PowerCabling` from the arm | reads the ECC values to decide inclusion, and biases the arm toward low dispersion on the exact dimension the study measures. Advised against |

**Why B is not available.** The binding constraint is not the target but the positivity floor of the log truncation rule, `min_q1_over_iqr`, which caps the parent's quartile ratio at `1 + 1/min_q1_over_iqr`: 3 at the current 0.5, against empirical quartile ratios of 4.5 to 284.6 in the five uncovered categories. Relaxing it to 0.01 still reaches a maximum sample coefficient of variation of only 2.15. Eight candidates were swept and none put a single synthetic dataset above 3. Reaching the empirical tail is a generator redesign.

**Why A is recommended.** The five datasets uncovered on dispersion are exactly
the categories the study already identifies as not one product and leaves whole
for that reason. The exception therefore falls where the paper has already told
the reader to expect trouble, and it can be written as one sentence that
strengthens the account rather than weakening it. The three uncovered on `n` are
decision 19 working as designed: the corpus stops at 9,999 values and the probe
set covers above it.


---

## New, found during Stage 2b

## 35. Physically implausible records were in the empirical arm

| | |
|---|---|
| **Manuscript** | Describes the empirical ECC datasets as the EPDs EC3 holds for each material category, cleaned by an interquartile rule. It does not say that any record was removed for being physically impossible, because none was. |
| **Measured in Stage 2b** | 115 of the 117,807 raw records reaching the arm declare a mass-based ECC above 100 kgCO2e per kg of product. The largest is a `RebarSteel` EPD at 2.59e6 kgCO2e/kg; two `Elevators` EPDs report 20,812 and 21,945; and 87 `Cement` records report a per-tonne GWP against a 1 kg declared unit, which is a factor-of-1,000 declaration error. |
| **Fix** | **Code, done in Stage 2b.** `empirical.MASS_ECC_CEILING = 100.0`, applied to mass-declared records only, before cleaning. Author decision, 2026-09-14. |
| **The framing the text must use** | **The bound is EXTERNAL and the paper has to say so.** This study measures the dispersion and modality of ECC distributions, so a ceiling read off the arm's own quantiles, standard deviations or visible gaps would be circular in exactly the way a dispersion-based category split would have been (decision 46). The bound comes from published embodied-carbon inventories, where the highest building-product coefficients are of order 13 kgCO2e/kg for primary aluminium (ICE v3.0), and is cross-checked stoichiometrically: 100 kgCO2e per kg of delivered product requires burning about 27 kg of pure carbon per kilogram shipped. It is set at 100 rather than at 25 so that it cannot be read as a tuned threshold, and it still catches the known cases by two orders of magnitude. **VERIFY THE ICE FIGURE against the source before it goes in the paper**; `refs/` holds no copy of ICE and the number above is from the analyst's knowledge, not from a document in this repository. |
| **Effect** | 11 of 117,090 cleaned values, 0.0094 percent, against a stop-and-report gate of 0.1 percent. No dataset lost; the arm stays at 149. Six datasets change: `Aggregates` unweighted coefficient of variation 13.601 to 6.424, `Chairs` 3.904 to 2.927, `Elevators` 3.159 to 1.838, `SteelSuspensionAssembly` 0.178 to 0.155, `Cement` 0.428 to 0.424, `AluminiumExtrusions` 0.842 to 0.818. The arm's maximum weighted coefficient of variation falls from 14.341 to 13.404 and its maximum excess kurtosis from 835.3 to 525.2. Arm-level medians move by at most 0.024 on any characteristic. |
| **What it does to the coverage claim, entry 34** | Improves it, without being aimed at it. Uncovered dataset-metric pairs fall from 11 of 1,490 to 10, and the uncovered-on-dispersion list from five datasets to four: `Elevators` is now inside the synthetic range. The remaining four are `PowerCabling` 13.40, `Aggregates` 7.11, `Grouting` 4.17 and `Chairs` 2.89 against a synthetic maximum of 2.58. |
| **A correction to entry 34 and to the canonical block** | Both attribute the arm's maximum coefficient of variation, 14.341, to `PowerCabling`. It was `Aggregates`; `PowerCabling` was second at 13.404. The list of uncovered categories was right, the attribution was not. |
| **Status** | Resolved in code. Text owes the rule, the external anchor, and the restated coverage list. |

## 36. The extraction discards carbon-negative products, by construction

| | |
|---|---|
| **Manuscript** | Reports the empirical arm as the EPDs EC3 holds for each category, with no statement that any part of the GWP range is excluded. |
| **Measured in Stage 2b** | The extraction keeps only records with a strictly positive ECC. That excludes **270 records across 57 of the 138 queried categories: 48 reporting exactly zero and 222 reporting a negative GWP.** The largest groups are `Carpet` 44, `Timber` 18, `DampproofingAndWaterproofing` 16, `BlanketInsulation` 16, `CMU` 13, `Insulation` 11. |
| **Why it matters** | Some of these are real. Biobased products can be legitimately carbon negative over a cradle-to-gate boundary that credits biogenic uptake, and the biobased categories are exactly where the negatives cluster: `Timber` 18, `WoodFlooring` 7, `MassTimber` 5, `NonStructuralWood` 4, `WoodDoors` 4, `CompositeLumber` 3, `HeavyTimber` 2, `WoodFraming` 1. Others are plainly errors, such as `SheathingPanels` at -12,105 kgCO2e/m3. The filter does not distinguish them. |
| **Consequence** | The empirical arm is truncated at zero by construction, so it cannot exhibit a left tail crossing zero and no conclusion about the lower tail of an ECC distribution generalizes to biobased products. This sits directly beside decision 13, which settles the support at (0, inf) open at zero: the two are consistent, but the paper currently states neither. |
| **Fix** | **Text, one or two sentences**, in the data section and beside the support statement. **This is a COUNT and not a change**: the filter is unaltered and the arm is unaffected. |
| **Status** | Open. Text. Measured in `audits/plausibility_ceiling.py`, table `outputs/tables/audits/TABLE_2b_NonPositiveGWP.csv`. |

## 37. The Methods text contradicts itself on the lognormal, and neither statement is what runs

| | |
|---|---|
| **Manuscript** | Says in one place that shape, location and scale are all estimated, and in another that location is held at zero. |
| **Code** | Neither. `customstats.weighted_lognorm_fit` returns `loc = 0.0` unconditionally -- its own docstring says "Location parameter (always 0 in this fit)" -- and optimizes over (sigma, mu) only. `fitting.fit_pewt_models` then builds `lognorm(s, loc = 0 - LOGFIT_OFFSET, scale)`. **The fitted threshold is a CONSTANT of -0.5, set by hand and never estimated.** The family in use is a three-parameter lognormal with two free parameters and a hand-set threshold. |
| **A consequence worth stating separately** | Because the threshold is never estimated, the unbounded-likelihood pathology cannot arise in the Stage 1 code: there is no optimizer over the threshold for it to break. The pathology is real and it is a property of the three-parameter fit the Methods text CLAIMS, not of the two-parameter fit the code performs. So the offset was not patching it. |
| **Fix** | **Both.** Code: Stage 2b replaces the fit; see entry 38. Text: state the family and the estimator that actually run, and state the parameter count plainly, because W1 is an in-sample criterion with no complexity penalty and the families differ in flexibility. |
| **Status** | Resolved in code by Stage 2b. Text owes a rewritten paragraph. |

## 38. A second finding in the same function: the "MLE" branch re-derives a closed form

| | |
|---|---|
| **Code** | `customstats.weighted_lognorm_fit` offers an "MLE" branch that hands `scipy.optimize.minimize` the weighted lognormal negative log-likelihood, and a "MoM" branch that computes the weighted mean and standard deviation of `log x`. For a lognormal those are THE SAME ESTIMATOR: the weighted MLE of (mu, sigma) is exactly the weighted mean and standard deviation in log space. The "MoM" label is a misnomer and the optimizer is re-deriving, numerically, a quantity available in closed form. |
| **Measured** | On `Cement` the two branches agree to 0.000e+00 in both parameters. |
| **Consequence** | Not a wrong number, but it is why `tests/fixtures` needed `rtol = 4.3e-08` on the lognormal W1 column while every other column agreed to 1.3e-14: the convergence path of the optimizer moves between scipy versions, so the only non-reproducible number in the whole fixture set came from an optimizer that was not needed. |
| **Fix** | **Code, done in Stage 2b.** `families.fit_lognorm2_mle` is the closed form, and the lognormal in production no longer calls the legacy function at all. |
| **Status** | Resolved in code. No text consequence beyond entry 37. |

## 39. The pLCA grouping silently dropped three datasets

| | |
|---|---|
| **Manuscript** | States 2,500 probabilistic LCAs over disjoint groups of four datasets drawn from 10,000. |
| **Code** | `corpus.make_combos` truncates with `ids[:len(ids) // nmats * nmats]`. The active corpus holds **9,999** datasets, not 10,000, because one parent failed to solve and was reported rather than approximated (decision 22 working as intended). So the grouping is **2,499 groups of four covering 9,996 datasets, and three datasets -- `dataset813`, `dataset2876`, `dataset7985` -- are in no pLCA at all.** Nothing said so. |
| **Fix** | **RESOLVED 2026-09-15 at source.** The corpus now holds 10,000 datasets (entry 48), so it divides by four and the grouping covers every one of them: **2,500 pLCAs, nothing held out**, which is what the manuscript already says. `corpus.describe_combos` still names any held-out datasets and both notebooks still print the line, so a future corpus that does not divide cannot fail silently. |
| **Text owed** | Nothing. "2,500 pLCAs" is correct again. |
| **Status** | Resolved. |

## 40. The lognormal is refitted, and it changes which method wins on the empirical arm

| | |
|---|---|
| **Manuscript** | Reports a lognormal fitted by maximum likelihood after offsetting the data by 0.5, and reports the KDE as the best-fitting method. |
| **What Stage 2b did** | Author instruction: implement the standard treatment of the unbounded three-parameter likelihood -- restrict the threshold to a closed interval bounded strictly below `min(x)`, maximize the remaining parameters at each grid point, take the INTERIOR local maximum of the profile likelihood. `families.fit_lognorm3_profile`. The +0.5 offset is gone. |
| **Which problem the offset was solving, measured** | **Near-zero values, not the threshold pathology**, and the pathology could not have been it: the Stage 1 code never estimates a threshold, so there is no optimizer for the divergence to break (entry 37). On the 14 of 149 datasets that still hold a value below 1 percent of their mean, the offset beats a no-offset two-parameter fit in 10 (71 percent), median gain 36.8 percent, mean W1 0.813 to 0.445. On the other 135 it wins 70 times of 135, a coin flip, median gain 1.4 percent. **Removing the near-zero values from the fit takes the no-offset mean W1 on those 14 from 0.813 to only 0.578**, so the offset does more than patch near-zeros: it also drags the family toward its normal limit, which is a crude fixed-value version of estimating the threshold. |
| **The pathology, demonstrated rather than asserted** | An unguarded joint optimizer over (threshold, mu, sigma) drives the threshold to within 1e-3 of a standard deviation of `min(x)` on **21 of the 149 empirical datasets (14.1 percent)**, with the fitted sigma reaching 11 to 24 against a guarded 2.0 to 2.5. Mean W1 0.248 unguarded against 0.182 guarded. |
| **How often the guard binds** | Empirical, over 149 datasets x 2 weightings: 124 interior (41.6 percent), **148 at the guard (49.7 percent)**, 26 at the normal limit (8.7 percent). Synthetic, 9,999 x 2: 11,469 interior (57.4 percent), 5,582 at the guard (27.9 percent), 2,947 at the normal limit (14.7 percent). **For about half the empirical arm the likelihood does not identify a threshold at all**, and it is set at a fixed fraction of a standard deviation below the smallest observation. `families.PROFILE_DELTA_LO_FRAC = 0.25`; see entry 43 for why 0.25 and not less, and for why the paper must describe it as a scale-aware version of the heuristic the offset was rather than as an estimate. Stage 2h sweeps it where it would have swept the offset. |
| **The two-parameter lognormal does NOT work, which was the simplest hoped-for answer** | Mean W1 on the empirical arm, variable weights: two-parameter 0.230, offset 0.183, profile three-parameter 0.178. The two-parameter fit is the WORST of the three and worse than gamma at 0.191. Dropping the offset without replacing it would have made the lognormal worse, not simpler. |
| **What moved** | Only the two Lognormal columns; every other column of `TABLE_EmpiricalECCMetricsAndW1.xlsx` is identical to 0.000e+00. Empirical `Lognormal, Variable` median W1 0.1367 to 0.1220 (-10.8 percent), mean 0.1785 to 0.1778; `Lognormal, Uniform` median 0.1647 to 0.1538, mean 0.2056 to 0.2106. Synthetic `Lognormal, Variable` median 0.0819 to 0.0713 (-13.0 percent), mean 0.1104 to 0.0988 (-10.5 percent). |
| **THE FINDING THAT MATTERS FOR THE PAPER'S CLAIM** | **The KDE is not the best method on the empirical arm and was not before this change either.** Mean rank over the six methods, 149 empirical datasets: `Lognormal, Variable` **2.09**, `KDE, Variable` 2.74, `Lognormal, Uniform` 3.34, `KDE, Uniform` 3.52, `Normal, Variable` 4.21, `Normal, Uniform` 5.11. Under the Stage 1 method the same two were 2.27 and 2.59, so refitting the lognormal widened its lead from 0.32 to 0.65. On the synthetic corpus the KDE still leads on rank, but by **0.06** (2.12 against 2.19, down from 0.42), and it has already lost the mean: `Lognormal, Variable` 0.0988 against `KDE, Variable` 0.1017. It keeps the synthetic median, 0.0635 against 0.0713. The paper cannot state "KDE fits best" without saying on which arm and by which summary. |
| **Status** | Resolved in code. **Text owes a rewritten lognormal paragraph, the parameter count, the guard, and a restatement of which method wins where.** See also entry 42, which is the stronger version of the same objection. |


## 41. The empirical arm was scored on UNNORMALIZED values

| | |
|---|---|
| **Manuscript** | Reports Wasserstein-1 distances for the empirical datasets alongside the synthetic ones, on a common axis, and reports means across the empirical arm. |
| **Measured in Stage 2b** | The Stage 1 empirical W1 column was computed on the RAW ECC values, not on the values divided by their unweighted mean, while the metrics table beside it in the same file WAS computed on normalized values. `WindTurbines` carries `Normal, Uniform` = 164.26 in `tests/fixtures/TABLE_EmpiricalECCMetricsAndW1.xlsx` against a `mean_uw` of exactly 1.0 in the adjacent column. Reproduced exactly: scoring the stored `dct_realeccs_trimmed.json` values as they sit gives 164.263, and 164.263 / 910.7 = 0.1804, which is the normalized value. The mean of that column across the arm is 9.38, against 0.48 now. |
| **Consequence** | Every empirical W1 MAGNITUDE was in the raw unit of its own category, so a concrete dataset at roughly 400 kgCO2e/m3 and a cement dataset at roughly 0.75 kgCO2e/kg were reported on the same axis three orders of magnitude apart for reasons that have nothing to do with fit quality. Any mean or median W1 across the empirical arm was effectively a magnitude-weighted average, and the empirical arm could not be compared with the synthetic arm at all. |
| **What it does NOT affect** | **The per-dataset RANKING of the six methods.** W1 is exactly linear in a rescaling of the data and all six models are fitted to the same values, so scaling multiplies all six scores by the same constant and the within-dataset order is untouched. Reported mean RANKS are therefore sound; reported W1 VALUES are not. |
| **Fix** | **Already fixed in code, as a side effect.** Stage 2a-2 rebuilt the empirical path so notebook 2 reads from `empirical.prepare`, which divides each dataset by its own unweighted mean. Nothing recorded that this also corrected the scale of the W1 column, which is why it is written down here. |
| **Status** | Resolved in code. **Text owes a replacement of every empirical W1 value**, and the new numbers are not a rescaling of the old ones by any single constant, so they cannot be converted -- they have to come from the rerun. Mean W1 across the 149 datasets is now `Normal, Uniform` 0.484, `Normal, Variable` 0.431, `Lognormal, Uniform` 0.206, `Lognormal, Variable` 0.178, `KDE, Uniform` 0.278, `KDE, Variable` 0.243. |

## 42. Every parametric family is fitted by likelihood and judged by W1, and it costs them the comparison

| | |
|---|---|
| **Manuscript** | Fits the normal and the lognormal by maximum likelihood, scores all six methods by Wasserstein-1, and concludes from those scores that kernel density estimation characterizes ECC datasets better than the parametric families. |
| **The objection** | Those are two different criteria. A family can lose a W1 comparison because it was never fitted under the rule it is judged by, and the KDE has no such handicap: it is not fitted by likelihood at all. A reviewer can raise this in one sentence, so Stage 2b measured it. |
| **What was done** | `fitting.fit_family(..., method='w1')` minimizes W1 directly over each family's parameters, starting from the maximum-likelihood fit so it can never score worse. Five families, both weightings, the full 149-dataset empirical arm and a 1,500-dataset synthetic sample. `audits/family_comparison.py`, `outputs/tables/audits/TABLE_2b_FamilyComparison.csv`. |
| **How much the mismatch was worth** | Median reduction in W1 from fitting by W1 instead of by likelihood, empirical arm: **normal 31.3 percent**, three-parameter lognormal 12.4, two-parameter 9.2, offset 8.3, gamma 7.7. It improves 85 to 93 percent of datasets in every family. The normal is hit hardest because the weighted mean and standard deviation are a poor W1 fit to a skewed sample. |
| **THE RESULT** | **On the empirical arm the KDE is beaten by every parametric family once they are fitted by the criterion they are judged by, and by three of five even under maximum likelihood.** Mean W1, variable weights: W1-optimal three-parameter lognormal 0.138, gamma 0.149, offset lognormal 0.151, two-parameter 0.153, normal 0.167, against the **KDE at 0.251**. Under maximum likelihood the order is three-parameter lognormal 0.178, offset 0.183, gamma 0.191, two-parameter 0.230, KDE 0.251, normal 0.436. |
| | **On the synthetic corpus the KDE no longer leads on the mean even under maximum likelihood.** Mean W1, variable weights: W1-optimal three-parameter lognormal 0.081, offset 0.096, two-parameter 0.098, gamma 0.099, maximum-likelihood three-parameter lognormal 0.100, **KDE 0.103**, normal 0.105. On the MEDIAN the KDE holds second place at 0.063, behind the W1-optimal three-parameter lognormal at 0.057 and ahead of its maximum-likelihood 0.072. |
| **How to read it** | The KDE's advantage is real on synthetic data drawn from smooth multi-component mixtures and is NOT real on the empirical arm. That is not a small caveat: the empirical arm is the one the paper's practitioner-facing conclusion applies to. |
| **Fix** | **Analysis and text.** The W1-optimal results should be reported alongside the maximum-likelihood ones rather than replacing them: maximum likelihood is what a practitioner would actually do, and W1-optimal fitting is the fair-comparison control that answers the objection. `FIT_METHOD = 'mle'` remains the study's method. |
| **A caveat the text must carry** | W1 is an IN-SAMPLE criterion with no complexity penalty, and the families differ in flexibility: normal 2 parameters, two-parameter lognormal 2, gamma 2, three-parameter lognormal 3, KDE effectively n. Fitting by W1 sharpens that, because the more flexible family gains more. Stage 2c owns the held-out or cross-validated answer and should do the same comparison there before any of this goes in the paper. |
| **Status** | Measured. **Needs an author decision on what to report**, and Stage 2c should repeat it out of sample. |

## 43. A model can score well on W1 and be unusable in the pLCA

| | |
|---|---|
| **Manuscript** | Selects among UQ methods by Wasserstein-1 distance between the fitted CDF and the weighted empirical CDF, and then uses the same fitted models as the sampling distributions of the probabilistic LCA. The two uses are never distinguished. |
| **Found in Stage 2b** | They are not the same requirement. The first version of the profile-likelihood lognormal, with the threshold guard at 0.01 of a standard deviation below `min(x)`, produced fitted models whose standard deviation reached **3,281 on the empirical arm and 5,345 on the synthetic one**, on data normalized to a mean of exactly 1 and a standard deviation near 0.6. Every W1 score looked reasonable. The defect surfaced only in the pLCA results: `eci_std` for `Lognormal, Uniform` went from a 99th percentile of 0.92 and a maximum of 1.56 to 33 and 532, and `eci_mean` reached 9.6. |
| **Why W1 does not see it** | W1 is the area between two CDFs. A thin far tail contributes almost nothing to that area however far it extends, so a model can match the body of the data, carry an arbitrarily heavy tail, and be scored as a good fit. A Monte Carlo that SAMPLES from the model is dominated by exactly that tail. |
| **Fix** | **Code, done in Stage 2b.** `families.PROFILE_DELTA_LO_FRAC = 0.25`, the smallest guard at which no fitted model on either arm has a standard deviation above five times the data's. It also improves mean W1 on both arms, so it costs nothing on the study's own criterion. `tests/test_families.py::test_profile_fit_is_a_usable_generative_distribution` asserts it. |
| **The general point, which outlives this particular fix** | **Every method in this study is selected by a CDF distance and then used as a sampler, and no part of the analysis currently checks that the selected model is a sane sampler.** The lognormal is where it bit because that family has an unbounded likelihood; nothing rules out the same failure elsewhere, and the KDE's own tail behavior under a bandwidth sweep is a Stage 2h question that has the same shape. |
| **Fix, the general version** | **Analysis, and it needs an owner.** Stage 2c owns the evaluation target and should decide whether W1 alone is a sufficient criterion or whether a tail-sensitive companion is needed; Stage 2g owns the downstream metrics and meets the same question from the other end. A cheap first step is to report, for every fitted model, the ratio of its own standard deviation to the data's. |
| **Status** | Fixed for the lognormal. **Open as a general question, owner unassigned, and it is the kind of thing a reviewer finds.** |

## 44. WHY the KDE loses on the empirical arm, and why two of the three reasons are defects in the comparison rather than facts about KDE

| | |
|---|---|
| **Context** | Entry 40 records that the KDE is not the best-fitting method on the 149 empirical datasets. That is a statement about numbers. This entry is the diagnosis, and it matters far more, because **two of the three mechanisms are defects in how the comparison is SPECIFIED.** `audits/why_kde_loses.py`. |
| **1. DATASET SIZE, and the two arms do not actually disagree** | The KDE's mean rank improves monotonically with n on BOTH arms and the lognormal's degrades. Empirical, `KDE, Variable` by size band: 3.60 at n = 3-9, 2.77 at 10-99, 2.54 at 100-999, **1.64 at n >= 1000**, where it is second of six and `KDE, Uniform` is first at 1.36. `Lognormal, Variable` runs the other way: 2.55, 1.76, 2.08, 3.64. The synthetic corpus shows the same pattern. **The corpus allocates 2,500 datasets to each size stratum, so 25 percent sit above n = 1,000, against 7.4 percent of the empirical arm** -- the empirical mix is 13.4 / 53.0 / 26.2 / 7.4 percent, median n = 50. Reweighting the corpus to the empirical size mix moves `KDE, Variable` from 2.12 to 2.25 and `Lognormal, Variable` from 2.19 to 2.02, **which is the empirical ordering**. The arms agree; equal allocation is a precision choice, not a claim about how common each size is. |
| **2. THE BANDWIDTH RULE, and this study does not use the author's own** | `BW_METHOD = 'scott'`, `1.06 * sd * n_eff ** -0.2`, oversmooths right-skewed data: median bandwidth / sd of **0.56** on the empirical arm under variable weights. A Gaussian KDE inflates the fitted variance by `sqrt(1 + (h/sd)^2)`, so that is 15 percent of spread the data does not have, and it is why the fitted KDE puts a mean of 9.4 percent of its mass below zero on an arm whose support is (0, inf). Silverman's robust rule, `0.9 * min(sd, IQR/1.34) * n_eff ** -0.2`, gives 0.33 and 5.2 percent. **Silverman beats Scott on 100 percent of the 149 datasets under variable weighting** (95.3 percent under uniform): mean W1 0.2507 to **0.1228**, against the lognormal's 0.1778. Under Silverman the KDE beats the lognormal on **81.2 percent** of datasets; under Scott, on 33.6 percent. **Torres et al. (2026), the author's KL2 paper, uses Silverman and justifies it explicitly. This paper uses Scott.** Entry 10 and decision 9 already flag the inconsistency; this is what it costs. |
| **3. THE CRITERION REWARDS UNDERSMOOTHING, which confounds mechanism 2** | W1 falls monotonically as the KDE bandwidth shrinks: mean W1 over the empirical arm at multiples of Scott's bandwidth is 0.2507 at 1.0, 0.1365 at 0.5, 0.0497 at 0.1, **0.0415 at 0.05**, 0.0444 at 0.02. The minimizing multiple is 0.02 for 95 of 149 datasets and 0.01 for 26 more; it turns up only at 0.01, and that is the 1,000-point scoring grid running out of resolution rather than a real optimum. A KDE with a vanishing bandwidth IS the empirical distribution it is being scored against. **So W1 cannot arbitrate between methods of different flexibility, and the Silverman result above is partly just "Silverman is smaller".** |
| **What survives the confound** | The direction, and it is the important part. Scott oversmooths this data badly, and **the KDE still loses to the lognormal under Scott even though the criterion is biased in the KDE's favor.** Both of those are real. What is NOT established is the magnitude of the KDE's disadvantage, or whether it has one at all under a defensible bandwidth and a criterion with a complexity penalty. |
| **Fix** | **Analysis, and it is already owned.** Stage 2h owns the bandwidth rule and must resolve the KL1 / KL2 / this-paper inconsistency. Stage 2c owns the evaluation target and must produce the out-of-sample or known-parent comparison, which is the only thing that can settle mechanism 3. **Neither is adopted here; nothing in this entry changes a default.** |
| **Status** | Measured, open, owners assigned. **This entry, not entry 40, is what the manuscript's discussion has to be built on.** |

## 45. The KDE bandwidth: where Silverman actually breaks, and the guarded rule that fixes it

| | |
|---|---|
| **Context** | Entry 44 measured that Silverman's rule beats Scott's on 100 percent of the empirical arm by W1, and that W1 is a biased referee because it rewards undersmoothing. The author's recollection was that Silverman "broke with small datasets when the IQR was unreasonably small". Both halves were checked. `audits/bandwidth_rules.py`. |
| **The referee** | **Leave-one-out likelihood cross-validation**, the standard bandwidth criterion (Habbema, Duin and Hermans 1974; Silverman 1986 section 3.4.4). It has a genuine interior optimum -- as h goes to zero each held-out point falls in the gap left by its own kernel and it goes to minus infinity -- so unlike W1 it penalizes a collapsed bandwidth. Used to compare Scott against Silverman, neither of which optimizes it. |
| **THE FAILURE IS NOT WHERE IT LOOKS.** | `(IQR/1.34)/sd` falls below 0.2 on 9 empirical fits, and those are large heavy-tailed categories -- `PowerCabling` at 0.008 with n = 400, `Aggregates` at 0.078 with n = 384, `Grouting` at 0.080 with n = 219. **On exactly those, Silverman beats Scott on held-out likelihood in 100 percent of cases**, mean advantage +1.20 nats. A tight core with extreme outliers is real structure and shrinking the bandwidth to fit it is correct. Silverman's `min(sd, IQR/1.34)` is doing its job, hardest, precisely where it looks most alarming. **Flooring the robust scale at sd/3 makes held-out likelihood WORSE** (-0.845 against -0.809), because it oversmooths those real tight cores. |
| **WHERE IT DOES BREAK: SMALL n** | Share of fits where Silverman beats Scott on held-out likelihood, empirical arm by size: **n 3-9, 12.5 percent**; 10-99, 40.5 percent; 100-999, **80.8 percent**; 1000 and above, 63.6 percent. Synthetic: 9.6, 28.4, 40.1, 53.7. The worst cases are all n = 3 to 14 -- `MetalStairs` n = 5 loses 5.47 nats, `BlownInsulation [cellulose]` n = 3 loses 5.24. On one synthetic dataset pure Silverman gives a non-finite held-out likelihood outright. |
| **Why** | The interquartile range is a consistent but INEFFICIENT estimator of scale: its asymptotic relative efficiency under normality is about 37 percent, so its standard error is roughly 1.6 times the sample standard deviation's, and at n = 3 to 10 the quartiles are interpolated between two order statistics. When that noisy estimate lands low the bandwidth collapses and the density becomes spikes. **The exactly-zero case fires on only 0.35 percent of fits, all at n <= 10 and all under variable weighting, and `weighted_bw` already guards it.** The damaging population is the near-miss cases just above zero, which nothing guarded. |
| **The fix** | `weighted_bw(..., bw_method='silverman_guarded')`: Silverman above an effective-sample-size threshold, the plain standard deviation below it -- NOT Scott, see entry 68. `SILVERMAN_MIN_NEFF = 30` at the time; **it is 20 from Stage 2c, entry 67 and decision 80**. It uses the KISH EFFECTIVE sample size, not raw n, because a concentrated Dirichlet draw can leave three effective observations in a 200-value dataset. |
| **Mean held-out log-likelihood, empirical arm** | always-Scott -0.773, always-Silverman -0.855, **guarded at 20 -0.725, at 30 -0.720, at 50 -0.728**. The guarded rule beats BOTH pure rules. It also repairs the worst cases: the 5th percentile of held-out log-likelihood goes from -2.053 under pure Silverman to -1.616. |
| **The threshold is calibrated on the referee, NOT on W1** | Deliberately, so that it is not tuned to the criterion the study then scores by. W1 still prefers pure Silverman (mean 0.151 against the guarded 0.178); that is the undersmoothing bias of W1 and is not evidence about the rule. |
| **What it does to the headline comparison** | **The KDE still beats every parametric family on both arms under the guarded rule.** Empirical, variable weighting, mean / median / 90th percentile / max W1: KDE-guarded 0.155 / 0.084 / 0.353 / 1.29 against three-parameter lognormal 0.178 / 0.122 / 0.371 / 1.58, gamma 0.191 / 0.130 / 0.387 / 2.59, two-parameter lognormal 0.230 / 0.148 / 0.475 / 3.97. Synthetic: KDE-guarded 0.081 / 0.039 / 0.213 / 1.04 against lognormal 0.100 / 0.071 / 0.216 / 0.65. |
| **Fix** | **Code, implemented and NOT adopted.** `BW_METHOD` is still `'scott'`; switching it is one constant and it moves every number in the paper. **Author decision, and Stage 2h formally owns the sweep.** |
| **Status** | Implemented, tested, measured. Awaiting an author decision on whether to adopt now or at 2h. |

## 46. Out of sample the KDE's advantage is a LARGE-DATASET advantage, and the weighting comparison stops being meaningful

| | |
|---|---|
| **Context** | Entries 42 and 44 record that the in-sample comparison cannot settle a contest between families of different flexibility. Stage 2b added held-out W1 -- fitted on half a dataset's values, scored against the other half, both directions, paired across all six methods -- as the control. `src/comparison.py`, notebook 2's final section. |
| **THE RESULT, and it is the same on both arms** | Held-out rank among the three estimation methods, within each weighting scheme, by dataset size. **Empirical**: n 10-99, lognormal 1.49 / normal 2.01 / KDE 2.49; n 100-999, lognormal 1.54 / KDE 1.62 / normal 2.85; **n >= 1000, KDE 1.18 / lognormal 1.91 / normal 2.91**. **Synthetic**: n 10-99, lognormal 1.67 / normal 1.69 / KDE 2.63; n 100-999, lognormal 1.57 / KDE 1.77 / normal 2.66; **n >= 1000, KDE 1.18 / lognormal 1.90 / normal 2.92**. |
| **How to state it** | **The KDE's advantage is real and it is a LARGE-DATASET advantage.** It survives out of sample, on both arms, above roughly n = 200 to 400, and it is decisive above n = 1,000. Below about n = 100 the lognormal wins out of sample, also on both arms. The in-sample comparison puts the crossover near n = 100; held out it moves right, which is what a complexity penalty is supposed to do to a flexible method. |
| **A CONFOUND THAT MUST NOT BE REPORTED AS A FINDING** | On the six-way held-out ranking, UNIFORM weighting appears to beat VARIABLE (empirical: `Lognormal, Uniform` 2.46, `KDE, Uniform` 2.85, `Lognormal, Variable` 3.05, `KDE, Variable` 4.00). **This is an artifact of the weights being an exchangeable Dirichlet draw and is not evidence about weighting.** The weights carry no information that generalizes from one half of a dataset to the other: under exchangeable weights the EXPECTED variable-weighted empirical CDF of a random half is the unweighted one, so a uniform-weighted fit is the better predictor of the held-out target by construction, and a variable-weighted fit is partly fitted to its own half's weight realization. |
| **What follows from that** | **Held-out scores may only be compared WITHIN a weighting scheme**, which is what `comparison.add_ranks(..., within_weighting=True)` does and what the figure shows. The in-sample comparison is unaffected: there the weights are a stated property of the dataset being described, not something being predicted. **The paper's weighting claim rests on the in-sample comparison and is untouched**; in sample, variable beats uniform within every family on both arms. This is discrepancy entry 32 seen from another angle, and it is a further argument for 2h's multiple-weight-realization work. |
| **Status** | Measured and implemented. **Text owes the size-conditioned statement of the KDE claim, and must not quote the six-way held-out ranking.** |

## 47. Why a normal can beat a KDE at small n, and how much of the score is the weight draw

| | |
|---|---|
| **Context** | The author refused two Stage 2b results on sniff-test grounds: that the lognormal beats the KDE below n = 1000, and that the NORMAL beats it below n = 100, with the objection "KDE converges to normal, so how would a normal ever beat it?" Both refusals were right to make. `audits/small_n_and_weight_noise.py`. |
| **THE ANSWER TO THE OBJECTION** | **A Gaussian KDE does not converge to the normal you would want.** It is the data convolved with a kernel, so its variance is the data's PLUS `h^2`: it converges to `N(mean, sd^2 + h^2)`, not `N(mean, sd^2)`. With a rule-of-thumb bandwidth `h = 1.06 * sd * n_eff ** -0.2` the standard-deviation inflation is **1.31x at n = 3, 1.20x at n = 10**, 1.135x at 30, 1.085x at 100, 1.035x at 1,000 and 1.014x at 10,000. The normal fit matches the standard deviation exactly by construction. So at small n the KDE is a systematically OVER-DISPERSED model and W1 charges it for that. |
| **Measured, and it tracks the theory** | Median (model sd) / (data sd) for `KDE, Variable` by band, empirical / synthetic: n 3-9 **1.201 / 1.266**; 10-99 1.103 / 1.142; 100-999 1.042 / 1.030; >= 1000 1.015 / 1.012. This is the bias-variance property of rule-of-thumb bandwidths, not a defect, and it is precisely why a KDE needs data. |
| **A CORRECTION to how the result was stated** | Ranked like for like -- among the three ESTIMATION methods only, within variable weighting -- **in sample the KDE is SECOND at n = 10-99, not last**: empirical lognormal 1.46, KDE 1.81, normal 2.73. It is last only below n = 10, where it is 2.65. The normal beats it only in the n = 3-9 band. The earlier six-way figures put `Normal, Variable` ahead of `KDE, Variable` at n = 10-99 on the HELD-OUT score, and **a held-out fit uses HALF the values, so that band is really measuring a KDE at n = 5 to 50**, which is exactly where the over-dispersion above is worst. The held-out penalty at small n is therefore partly an artifact of halving n, and it falls hardest on the method most sensitive to n. |
| **THE SECOND OBJECTION, and it is half right in the half that matters** | The author: "The values are pulled from a parent distribution as though they're uniform, but then fit with Dirichlet weights, so the weighted dataset doesn't really align with the parent." On the SYNTHETIC arm `genconfig.mode_coupling = 1.0`, so market share attaches at the MODE level (decision 21) and the parent has a market-weighted version, `MixtureParent.cdf(scheme='market')` -- the weighted data IS a sample from a real population. But the weights WITHIN a mode are a flat Dirichlet, and on the EMPIRICAL arm the weights are a flat Dirichlet throughout with no market information at all. |
| **So the scoring target has a noise floor, and it was never measured** | W1 between the SAME values under two independent Dirichlet draws, empirical arm, median by band: **n 3-9 0.2100; n 10-99 0.1499; n 100-999 0.0784; n >= 1000 0.0061.** Overall median **0.1344**, against the best method's median W1 of **0.0984**. **The target's own noise floor is larger than the best method's score.** |
| **What that does and does not invalidate** | It does NOT invalidate the aggregate: over 149 datasets the draw averages out, and across five independent weight realizations the mean ranks move by at most 0.12 and places three to six never change. It DOES mean **size-banded claims below about n = 100 are not resolvable at the precision they were quoted to**, because the floor there (0.15) exceeds the gaps between methods. |
| **And it changes how the headline should be stated** | `KDE, Variable` and `Lognormal, Variable` are within the draw noise of each other on MEAN RANK on the empirical arm -- 2.243 against 2.353 averaged over five realizations, and the lognormal edges ahead in one of the five. On WIN SHARE the KDE leads in every realization, 39 to 45 percent of datasets against 25 to 31 percent. **State the empirical result as a win share, not as a mean rank.** |
| **Fix** | **Text and analysis.** Text: state the KDE claim as size-conditioned and as a win share; state the over-dispersion as the mechanism, because it is the honest reason a KDE needs data and it strengthens rather than weakens the account. Analysis: **Stage 2h should average over weight realizations**, which cuts this floor by sqrt(K), and **Stage 2c should score the synthetic arm against `scheme='market'`**, which has NO draw noise at all and is the clean fix. |
| **Status** | Measured. Text owes the restatement; 2h and 2c own the analysis. |

**A FOURTH POINT, and it is a defect in the READOUT rather than in the analysis.**
The author pressed on the n < 10 result a third time and was right to. Scored --
always, and this is worth stating because it was asked twice -- **against the
weighted empirical CDF of the data itself, never against the parent**. So the
small-n ranking is not an artifact of an inapplicable reference.

What it IS: a real but negligible ordering, reported through a readout that
hides how small it is. The median relative gap between the best and worst of the
three estimation methods on the empirical arm is **0.21 at n = 3-9** and **7.21
at n >= 1000**. A rank turns both into "1, 2, 3". Mean W1 divided by the mean
across methods, empirical arm:

| band | KDE, Var | Lognormal, Var | Normal, Var |
|---|---|---|---|
| n 3-9 | 0.907 | 0.864 | **0.842** |
| n 10-99 | 0.782 | **0.714** | 1.162 |
| n 100-999 | **0.407** | 0.687 | 1.715 |
| n >= 1000 | **0.256** | 0.820 | 1.885 |

At n = 3-9 all three sit between 0.84 and 0.91 of the dataset mean: **they are
the same to within about 8 percent, and calling one of them "last" is reporting
noise-scale structure as a result.** At n >= 1000 the KDE is at 0.256 against the
normal's 1.885, a factor of seven. The mechanism behind the small-n ordering is
still the over-dispersion above -- and at n = 3-9 the profile lognormal is AT its
normal limit in 40 percent of fits and at its guard in another 40, so two of the
three "methods" are the same object there.

**Fix, done:** `comparison.add_relative` and a second row in
`CompareUQMethods_FIG_RankVsDatasetSize.png` showing the size of the gap beside
the rank. **The text must not state a winner below about n = 100 without the
relative figure beside it.**

## 48. One synthetic dataset is missing because a rejected draw was treated as a failure

| | |
|---|---|
| **Manuscript** | States 10,000 synthetic datasets, 2,500 in each of four size strata. |
| **What is there** | **9,999**, with the second stratum holding 2,499. `dataset4647`, n = 16, was never written. |
| **Why, reproduced exactly** | Replaying generation from the corpus seed: the draw failed with `component_targets_exhausted`, and all twelve component-solve attempts returned `unbounded_density` -- the moment targets drawn for one mixture component could only be met by a J-shaped beta or beta-prime, which is refused because such a component has infinite density at an endpoint and is drawn as a spike. |
| **THE DEFECT** | `generator.generate_dataset` retries a failed parent draw up to twenty times, but only when the status is `mode_too_narrow`. Any other status breaks out immediately, under a comment reading "a real failure, not a rejected draw: do not retry". **`component_targets_exhausted` IS a rejected draw**: the targets are themselves random, and drawing new ones would almost certainly have succeeded. `corpus.generate_corpus` then skips the slot rather than refilling it, and the failure REASON is recorded nowhere -- `runmeta.json` carries the count and `invalid_datasets.json` stays empty. |
| **What it is not** | It is not the principle that a target which cannot be met is reported rather than approximated. That principle is about not fudging a target; it does not require abandoning the slot. |
| **Fix** | **Code, one line**: retry on `component_targets_exhausted` as well, and record the reason for any slot that is finally abandoned. The corpus would then hold exactly 10,000 and the pLCA would divide by four, which removes the three held-out datasets of entry 39 as well. |
| **Cost** | **It requires regenerating the corpus**, which moves every downstream number. That is an author decision; generation is otherwise settled. |
| **Status** | **RESOLVED 2026-09-15.** The author authorized the regeneration. `generator.REDRAWABLE` now names both rejection statuses, `corpus.generate_corpus` records the reason for any slot it does abandon, and `corpus_2026-09-15` holds **10,000 datasets, 0 failed parents**. The pLCA is now 2,500 groups covering all 10,000, so entry 39's three held-out datasets are gone too. **The aggregates barely moved**, which is the check that the generator is stationary: mean W1 by method changes in the fourth decimal (KDE/Variable 0.0778 to 0.0776), mean rank by at most 0.007, median coefficient of variation 0.5018 to 0.5032. The empirical arm is bit-identical. |

## 49. The weight of statistical outliers was reported as zero whenever one value carried a quarter of the weight

| | |
|---|---|
| **Manuscript** | Reports "weight of statistical outliers" as one of the dataset characteristics, defined as the weight of values farther than 1.5 x IQR outside the interquartile range. |
| **What was there** | The quartiles were interpolated on `weighted_ecdf`'s PADDED arrays. That function prepends `-inf` and appends `+inf` so the interpolator it returns extrapolates flat outside the data, and those sentinels are not data. Whenever the smallest value carried more than a quarter of the weight, 0.25 fell in the padded first segment and `q1` came back as `-inf`. The interquartile range was then infinite, `q1 - 1.5*IQR` was `-inf` and `q3 + 1.5*IQR` was `+inf`, so both comparisons were False and **the characteristic was reported as exactly zero**. Where both quartiles landed in the padding the subtraction produced NaN. |
| **Who it hit** | Small datasets, almost entirely. The largest affected dataset has n = 39; most have n = 3. A flat Dirichlet draw over three points puts more than a quarter of the weight on the smallest one often. |
| **Fix** | Interpolate on the interior of the ECDF, which clamps to the smallest observed value rather than to a sentinel. One line, in `customstats.empirical_metadata`. |
| **What moved** | `synthetic weight_outliers` 474/10,000 rows, mean 0.0540 -> 0.0609. `synthetic weight_outliers_uw` 163/10,000, 0.0526 -> 0.0580. `empirical weight_outliers` 3/149, 0.0614 -> 0.0656. `empirical weight_outliers_uw` 1/149, 0.0590 -> 0.0612. **No W1 column moves on either arm**, and no other characteristic moves. |
| **What it does NOT change** | The generator calibration. Both arms are corrected by the same code and move the same way by a similar amount, so the arm-to-arm agreement the tuning objective measures is essentially unchanged. Generation stays closed. |
| **For the text** | Any reported mean, range or figure involving weight of outliers must come from the rebuilt tables. If the paper states that some datasets have no outlier weight, check it: part of that population was an artifact of this defect. |
| **Status** | **RESOLVED 2026-09-15.** Decisions 57 and 58. The synthetic arm needed `corpus.remetric_corpus` to pick the correction up, because it stores its characteristics at generation time rather than recomputing them; that is a recomputation of the readout and not a regeneration, and `values.parquet` is byte-identical across it. |

## 50. Three strip figures were redrawn differently on every run of the same table

| | |
|---|---|
| **Manuscript** | Uses the KS, W1 and W2 strip-and-rank figures as supplementary evidence. |
| **What was there** | All three called `sns.stripplot(..., jitter=0.4)`. Seaborn draws that jitter from the GLOBAL numpy random state, which the project's standing constraints forbid: "All randomness comes from an explicitly passed Generator, never from global numpy state." Two runs of the same notebook on identical tables produced different figures. |
| **How it was found** | The author re-ran notebook 2 to review it. Every number reproduced exactly and exactly three figures changed. |
| **Severity** | Cosmetic in effect -- only the vertical scatter of the points moves, and no number, rank or axis changes -- but it breaks the property that a figure is reproducible from the table behind it, which is a claim the deposit makes. |
| **Fix** | `comparison.rank_strip` takes a Generator, draws the offsets itself and returns one handle per rank so the legend still builds. Tested for reproducibility under a fixed seed, for sensitivity to a different seed, and for leaving the global stream untouched. |
| **Status** | **RESOLVED 2026-09-15.** No number moved. |

## 51. Two EC3 categories were not one product population, and a relative outlier filter could never have found that

| | |
|---|---|
| **Manuscript** | Reports the empirical arm as a set of material categories, each standing for one material choice in a pLCA. |
| **What was there** | `Chairs` held 86 records of which only 15 name any kind of seating; the rest are kitchen mixer taps, asphalt, culverts, hollowcore slabs, particle board and bathroom furniture. `Grouting` held 225 of which only 44 name a grout; the rest are gypsum plasters, decorative renders, ground granulated blast furnace slag, concrete admixtures, epoxy coatings, a cable clamp and a glazed door. `Aggregates` held 7 finished products -- sinks, washbasins, porcelain stoneware slabs -- among 385 records of crushed stone, gravel and sand. |
| **Why the cleaning did not catch it** | **This is the important part, and it is a general point about relative filters.** The symmetric log-space 3 x IQR rule works when a category is coherent: on `ReadyMix [4000-4999 psi]` its upper bound sits at 3x the median and it trims 42 records. On a contaminated category the same rule is inert, because the width of the interval is set by the spread of the very contamination it is supposed to remove. Measured upper bound as a multiple of the median: `ReadyMix` 3x, `Grouting` 289x, `PowerCabling` 88,518x, `Chairs` 18,469,729x, `Aggregates` 41,238,610x. The last three trimmed NOTHING. |
| **Fix** | Drop the two categories that are not one population; exclude the named intruders from the one that is. Both read the product NAME and never the ECC value, which is the constraint decision 43 sets and the same evidence the insulation split already uses. Decisions 60 and 61. |
| **An exclusion list, not an inclusion vocabulary** | An inclusion rule has to anticipate every legitimate naming convention in every language. Tried first, it would have removed 22 `PowerCabling` records that are plainly cables (`Cable a Haute Tension`, `TSLF 24kV`, `NF C 33-226`, `Nexans U-1000 R2V`, `H07RN-F`) and two Schindler elevators named by model number, while missing the actual intruders, since `GRANITEK Sinks` matches on "granite". |
| **What moved** | 149 datasets to **147**; 117,079 ECC values to **116,768**. Generator calibration IMPROVED and stayed far inside the gate: weighted objective 0.2308 to 0.2278, 0.45 of the 0.0066 seed-to-seed standard deviation. No regeneration. |
| **For the text** | The arm is 147 datasets and the derivation in the canonical block above is the one to quote. `Chairs` and `Grouting` should be named in the limitations as categories EC3 labels for a product but populates with a mixture. |
| **Status** | **RESOLVED 2026-09-16.** |

## 52. The canonical length unit was the inch

| | |
|---|---|
| **Manuscript** | Reports ECC in kgCO2e per declared unit, in an otherwise metric analysis. |
| **What was there** | `funcs_unit_conversion` converted every length-declared product to kgCO2e per INCH. `PowerCabling` and `DataCabling` were therefore reported per inch, and `PowerCabling`'s median read 0.0599 where the metric figure is 2.36 kgCO2e/m. |
| **Fix** | `length2m`. The frozen extract stores `ecc` already computed and cannot be rebuilt -- the EC3 store's sha256 no longer matches, so a rebuild would change the population, which decision 44 refuses -- so `empirical.fix_length_unit` rescales the 1,500 length-declared rows at load. |
| **What moved** | **No analysis number.** Every dataset is normalized by its own unweighted mean and cleaned by a log-space rule, both invariant under a constant rescale; the fixture regression passes unchanged. Only the magnitude and the label of the raw ECC change. |
| **Kept imperial, deliberately** | `psi` for concrete strength, which is what decision 46's strength classes are named in and what a US structural engineer specifies, and `rval` for thermal resistance. |
| **Status** | **RESOLVED 2026-09-16.** Decision 62. |

---

## New, found during Stage 2c

## 53. The evaluation target was the training data, and the paper has to say what replaces it

| | |
|---|---|
| **Manuscript** | Describes goodness of fit as the Wasserstein-1 distance between each fitted model's CDF and the variable-weighted empirical CDF of the dataset. |
| **Code** | That is exactly what it did, for every method, both arms, through Stage 2b. |
| **Why it is a problem** | Two reasons and neither is small. **(a) It is the training data.** A flexible method scored on its own training data is rewarded for flexibility with no complexity penalty, and the KDE is the most flexible method under test by a wide margin: in-sample W1 falls monotonically as its bandwidth shrinks, to about 2 percent of any standard rule, because a KDE with a vanishing bandwidth IS the empirical distribution it is being compared with. **(b) It is the variable-weighted CDF**, which makes "variable weighting improves fit" close to true by construction, since that CDF is itself the target. |
| **Fix** | **Analysis, done in Stage 2c, and it needs a Methods paragraph.** Two non-circular targets, one per arm. SYNTHETIC: W1 against the KNOWN PARENT, recovered exactly for all 10,050 datasets by replaying the generator (`corpus.rebuild_parents`). EMPIRICAL: cross-validated W1, ten random half-splits in both directions, all six methods sharing each split so the comparison is paired. Both are reported beside the in-sample score, not instead of it. |
| **Status** | Open. **Methods text plus every results number.** `src/recovery.py`, notebook 2's final section, `outputs/tables/TABLE_TargetComparison.csv`. |

## 54. What the new targets do to the headline, and the two arms disagree

| | |
|---|---|
| **Numbers** | Synthetic arm, mean W1. In sample: `KDE, Variable` 0.0776 best, `Normal, Uniform` 0.2188 worst. **Against the parent: `KDE, Uniform` 0.1228 best and `KDE, Variable` falls to third at 0.1639.** Win share moves from `KDE, Variable` 0.706 in sample to `KDE, Uniform` 0.515 against the parent. Empirical arm, cross-validated: `Lognormal, Uniform` **0.2908** best, `Lognormal, Variable` 0.3088, `KDE, Uniform` 0.3214, `KDE, Variable` fourth at 0.3411, against an in-sample ordering that put `KDE, Variable` first at 0.1471. Within a weighting scheme the lognormal's cross-validated win share is 0.496 (uniform) and 0.598 (variable) against the KDE's 0.331 and 0.197. |
| **The disagreement** | Out of sample the two arms give **different answers about the family**, and both differences survive a paired bootstrap over datasets. Against the parent the KDE beats the lognormal by +0.0078 [+0.0064, +0.0092] (uniform) and +0.0060 [+0.0043, +0.0076] (variable), winning 72.1 and 70.2 percent of datasets. Cross-validated on the empirical arm the lognormal beats the KDE by 0.0306 [-0.0479, -0.0138] and 0.0323 [-0.0495, -0.0156], with the KDE winning only 41.7 and 32.3 percent. |
| **Why, and it is not a contradiction** | Removing one difference at a time, uniform weighting: parent, equal allocation **+0.0078**; the same corpus CROSS-VALIDATED instead **-0.0034**, because a cross-validation half measures the KDE at n/2 and its advantage is a large-n advantage; reweighted to the empirical size mix **-0.0127**; the empirical arm itself **-0.0321**. **The criterion and the size mix account for the SIGN.** A factor of about two in magnitude does not, and that is a genuine corpus-to-arm difference rather than an artifact of how either is read. |
| **What every arm and criterion agrees on** | The SHAPE. The KDE loses to the lognormal at n = 10-99 and wins at n >= 1000, on the parent, cross-validated on either arm, and in sample. |
| **Fix** | **Text.** The paper cannot state a single winner. It must state the size dependence, report both arms on their own non-circular criterion, and say which criterion and which size mix each number comes from. |
| **A note on which numbers to quote** | The SYNTHETIC scores are deterministic. The cross-validated ones move in the third decimal with the random splits: notebook 2 and `audits/evaluation_target.py` use independent streams and read 0.2908 against 0.2910, 0.3088 against 0.3132, 0.3214 against 0.3223, 0.3411 against 0.3467. **Quote notebook 2's**, which are in `TABLE_TargetComparison.csv`. No ordering and no conclusion differs between them. |
| **Status** | Open. **This is the paper's central claim and it is now conditioned rather than settled.** `audits/evaluation_target.py` section 3c reproduces the reconciliation, which is one coherent computation from that script and so uses its own split stream throughout. |

## 55. "Variable weighting improves fit" was true by construction, and on a common target it is a size effect

| | |
|---|---|
| **Manuscript** | Reports that variable weighting improves goodness of fit. |
| **Why it was circular** | Every model, including the three uniform-weighted ones, is scored against the VARIABLE-weighted empirical CDF. A uniform-weighted model is therefore charged a distance no estimation method can remove. Decomposed: on the empirical arm the definitional term is **0.1119**, identical for all three uniform-weighted methods, and it is **61.9 percent** of `KDE, Uniform`'s total score and 56.2 percent of `Lognormal, Uniform`'s. On the synthetic arm it is 0.1373, and 85.2 percent of `KDE, Uniform`'s total. **Ranking the six against their OWN weighting scheme reverses the conclusion**: `KDE, Uniform` goes from rank 3 to rank 1 and `KDE, Variable` from 1 to 2, on both arms. |
| **The uncircular answer** | On the synthetic arm both weighting schemes can be scored against ONE target, the market-weighted parent, which is the population a pLCA of what gets built is a statement about. There a uniform-weighted model pays a BIAS and a variable-weighted model pays VARIANCE from the noisy Dirichlet weights. **Overall it is a coin flip**: variable weighting wins on 51.1 percent of datasets for the normal, 54.6 for the lognormal, 54.2 for the KDE, and the paired bootstrap interval straddles zero for the lognormal and the KDE. **It is strongly size dependent and distinguishable in every band**: for the KDE, variable weighting is worse by 0.0395 at n = 3-9 and better by 0.0571 at n >= 1000; it wins 39.0 percent of datasets in the smallest band and 75.8 percent in the largest. The lognormal shows the same pattern more strongly, -0.0580 to +0.0451, winning 33.6 percent then 77.7. The two parents are 0.0828 apart on average before any fitting, which is the floor a uniform-weighted method cannot beat. |
| **The empirical arm cannot answer this** | Its weights are an exchangeable flat Dirichlet draw with no market information, so the expected variable-weighted CDF of a random half IS the unweighted one and a uniform-weighted fit is the better cross-validated predictor by construction. That is a property of the synthetic weights, not a finding about weighting. **A cross-validated score may be compared across estimation methods within one weighting scheme and never across weighting schemes.** |
| **Fix** | **Text, and it changes a claim.** Variable weighting does not improve fit in general; it helps when there are enough data points to estimate the reweighted distribution and hurts when there are not, with the crossover around n = 100. |
| **Status** | Open. `TABLE_WeightingDecomposition.csv`, `TABLE_WeightingOnCommonTarget.csv`. |

## 56. Regret, which is the number a practitioner needs and the paper does not report

| | |
|---|---|
| **Why** | A win share answers "how often is this method best". A practitioner will use ONE method on every dataset, so the question is what that costs when it is not the best one, and a method that is second by a hair everywhere is a better default than one that wins half the time and is catastrophic on the rest. |
| **Numbers, synthetic arm against the parent** | Mean regret: `KDE, Uniform` 0.0290, `Lognormal, Uniform` 0.0368, `KDE, Variable` 0.0702, `Lognormal, Variable` 0.0762, `Normal, Uniform` 0.1166, `Normal, Variable` 0.1431. **The upper tail reverses two of them**: at the 95th percentile `Lognormal, Uniform` is 0.1281 against `KDE, Uniform`'s 0.1506, and the worst case is 0.86 against 1.26. `KDE, Uniform` is the best method on 51.6 percent of datasets and the lognormal on 16.4. |
| **Numbers, empirical arm cross-validated** | Mean regret: `Lognormal, Uniform` **0.0244**, `Lognormal, Variable` 0.0425, `KDE, Uniform` 0.0550, `KDE, Variable` 0.0748, then `Normal, Variable` 0.2075 and `Normal, Uniform` 0.2147. `Lognormal, Uniform` is the best method on 33.1 percent of datasets and has the tightest p95, 0.0953 against `KDE, Uniform`'s 0.2411. |
| **Fix** | **Text.** Report the regret distribution, not only the win rate. The sentence the numbers support is that the KDE has the lower mean cost and the lognormal the tighter worst case. |
| **Status** | Open. `TABLE_Regret.csv`, and `CompareUQMethods_FIG_Regret.png`. |

## 57. The KDE's advantage is NOT about multimodality

| | |
|---|---|
| **The assumption** | The natural reading of a KDE-versus-parametric comparison is that the KDE wins where the data are multimodal. Stage 2a-2 already made that doubtful by measuring 95 percent of empirical datasets at exactly ONE visible mode. |
| **What was measured** | Within each size band, on the synthetic arm against the parent, at n >= 8 so that `modality.n_modes_visible` has actually measured something. **The split by modality barely moves the answer and the split by SIZE decides it.** `KDE, Uniform` mean rank at n = 100-999: 1.57 on visibly unimodal datasets and 1.47 on multimodal ones. At n >= 1000: 1.23 and 1.25. The KDE's win share against the lognormal at the same weighting, on visibly UNIMODAL datasets, runs 48 percent at n = 10-99, 82 at n = 100-999 and 99 at n >= 1000. |
| **Where the advantage comes from instead** | The location/shape split says it plainly. At n >= 1000 on visibly unimodal datasets, variable weighting: the lognormal gets the MEAN slightly better, 0.0159 against the KDE's 0.0173, and loses on SHAPE by more than a factor of two, 0.0569 against 0.0243. **The KDE's advantage in the unimodal majority is a shape advantage** -- skewness and tail behavior a two- or three-parameter family cannot match -- and not an ability to represent several modes. |
| **Fix** | **Text, and it is a different claim from the one the paper is set up to make.** Do not attribute the KDE's performance to multimodality. `TABLE_ModalityConditioned.csv`. |
| **Status** | Open. |

## 58. Overlap area agrees with W1, which is what justifies keeping W1

| | |
|---|---|
| **Why it was asked** | CLAUDE.md names Prado-Lopez et al. (2014) and asks for overlap area alongside W1 as a robustness check, so the choice of W1 can be defended in the paper. |
| **Result** | On the synthetic arm, where a reference DENSITY exists, overlap area and W1 pick the same winner on **66.6 percent** of datasets, their per-dataset scores correlate at Spearman **0.689**, and **they give the same mean-rank ordering of all six methods**: `KDE, Uniform` 2.03 by W1 and 1.82 by overlap, then `KDE, Variable`, the two lognormals, the two normals, in the same sequence. |
| **The limitation, and it is the reason W1 stays the criterion** | Overlap area needs a reference density, so it cannot be computed on the empirical arm at all: the target there is a set of atoms, and supplying a density would mean choosing a bin width or a kernel -- and a kernel would score the KDE against a KDE. W1 compares CDFs and needs nothing. |
| **Fix** | **Text, one or two sentences.** State that the criterion was checked against the main alternative in the comparative-LCA literature on the arm where both are computable, that they agree on the ordering, and that overlap area was not adopted because it is undefined against an empirical target. |
| **Status** | Open, and the carried-forward item is **RESOLVED**. |

## 59. The three-parameter lognormal is indistinguishable from gamma on real data

| | |
|---|---|
| **Why it matters** | Decision 51 records that the profile-likelihood guard, not the data, sets the lognormal's threshold for 51.2 percent of the empirical arm. A reviewer will ask why a family with an unbounded likelihood, a pathology and a guard is used when gamma has none of the three. Stage 2b's own assessment named this as the stage's honest risk. |
| **Result, out of sample** | On the empirical arm, cross-validated, the three-parameter lognormal is **indistinguishable from gamma, from the two-parameter lognormal, and from the Stage 1 offset method**: every paired bootstrap interval straddles zero. Uniform weighting, mean paired difference against gamma +0.0044, CI [-0.0025, +0.0119]; variable +0.0008, CI [-0.0055, +0.0077]. Only the normal separates. On the synthetic arm against the parent it DOES separate from gamma, +0.0117 uniform and +0.0045 variable, winning 77.4 and 67.5 percent of datasets. |
| **And Stage 2b's specific claim is withdrawn** | Stage 2b reported that gamma beats the three-parameter lognormal on the datasets where the guard binds. Out of sample it does not: on those 65 empirical datasets gamma wins **47.7 percent**, a coin flip, and on the `interior` datasets -- where the likelihood does identify a threshold -- gamma wins 61.4 percent, which is the opposite direction. |
| **Fix** | **Text.** Keep the three-parameter lognormal: it is never worse and it is better on the synthetic arm. State plainly that on real ECC data its third parameter buys nothing measurable, and that gamma would have served as well. Do NOT build a hybrid estimator; the measurements favor it least. |
| **Status** | Open. `audits/family_out_of_sample.py`, `TABLE_FamilyOutOfSample.csv`. |

## 60. The bandwidth, refereed by the parent: Scott is confirmed wrong and the guard is not confirmed

| | |
|---|---|
| **Why a third referee** | Decision 54 chose the guarded Silverman rule on leave-one-out likelihood, because in-sample W1 cannot arbitrate a bandwidth: it falls monotonically as the bandwidth shrinks. But leave-one-out likelihood is a DENSITY criterion and the study scores CDFs, so the setting was chosen on one criterion and reported on another. The synthetic parent removes that objection: W1 against the parent is the study's own criterion measured against something that is not the training data. |
| **The referee is unbiased, which had to be checked first** | Against the parent only **1.2 percent** of datasets put the optimal bandwidth at the sweep floor of 0.02 of Scott's, against in-sample W1 minimizing there for 95 of 147. It has a genuine interior optimum, at a median of **0.46 (uniform) and 0.56 (variable) of Scott's**. |
| **What it confirms** | **Scott oversmooths.** It sits 1.386x (uniform) and 1.349x (variable) above the parent-optimal bandwidth, and the guarded rule beats it on **72.1 and 66.4 percent** of datasets. The direction of decision 54 is right. |
| **What it does NOT confirm** | The GUARD. Pure Silverman beats the guarded rule on 90.1 and 83.2 percent of datasets against the parent, and is the best of the three named rules on 69.5 and 64.1 percent. **The guard's cost is small** -- mean W1 against the parent 0.1180 against pure Silverman's 0.1169 (uniform) and 0.1597 against 0.1593 (variable), which is 0.9 and 0.25 percent -- and it buys the repaired p05 of held-out likelihood that decision 54 was chosen for. Stated and left alone; changing the guard is an author decision. |
| **The reconciliation, and it belongs in the text** | A density criterion and a CDF criterion want different bandwidths. The empirical CDF is already root-n consistent, so smoothing buys a CDF criterion very little, while a density criterion punishes undersmoothing hard. That is why W1 prefers a smaller bandwidth than leave-one-out likelihood does, on the parent as well as in sample, and it is the honest explanation rather than "W1 is biased". |
| **Fix** | **Text.** Present the bandwidth in the order the evidence came: Scott was in use, the KDE lost under it, Scott turned out to oversmooth on three independent criteria, and the rule the author's own KL2 paper defends was adopted. Report Scott as a sensitivity throughout, not as a discarded option. |
| **Status** | Open. `audits/bandwidth_against_parent.py`, `TABLE_BandwidthAgainstParent.csv`. |

## 61. The 1,000-point scoring grid, measured at last, and it does not reach the conclusions

| | |
|---|---|
| **The question** | Stage 2b handed this over as its third item for Stage 2c: the scoring grid is 1,000 equally spaced points, so its resolution near zero is the same for every dataset and is coarse on one spanning orders of magnitude. Nothing had measured what it costs. |
| **The level** | Against a 200,001-point lattice on the same interval the study's criterion is off by a median of 0.20 percent on the empirical arm and 0.11 percent on the synthetic, with a p99 of about 14.5 percent on both. It picks a different winner on **1.36 percent** of empirical and **0.50 percent** of synthetic datasets. |
| **It is biased BY METHOD, and against the KDE** | Mean W1 on the empirical arm, coarse against dense: `KDE, Variable` 0.1471 against 0.1406, a **+4.7 percent** bias, and `KDE, Uniform` +2.9 percent, against +0.2 percent for both lognormals. The KDE's CDF has the most structure at the scale of a grid cell, and it is the method under test. |
| **It does not reach a conclusion, and that is what decides it** | The discretization is common to all six methods on a given dataset, so it moves the LEVEL of every score and not the gap between two of them. The paired cross-validated KDE-minus-lognormal difference on the empirical arm is **-0.0340 at the study's grid and route, -0.0341 integrating the same grid as two CDFs, and -0.0339 at 20,000 points**. It cancels. |
| **SUPERSEDED, and the first reading was wrong twice** | This entry said the trapezoid route was a free improvement not being taken, and that the discretization cancels. Both need correcting. **It is not simply better**: at 1,000 points the atom route is better TYPICALLY, median 0.00134 against 0.00218, because the data's empirical CDF is a step function that a discrete-to-discrete distance handles exactly. And it cancels only in the CROSS-VALIDATED comparison; the IN-SAMPLE paired difference moves from -0.0184 to -0.0229 under uniform weighting. |
| **What actually decided it** | **The atom route never converges.** Adding points does not extend the grid, whose top is `max(x) + 10 sd` whatever the count, so a model with mass beyond it keeps losing that mass: p99 relative error sticks at 0.0379 from 20,000 points through 100,000 while trapezoid goes 0.0039 to 0.0010 to 0.0002. |
| **Fix** | **Analysis, DONE.** 20,000 points and trapezoid quadrature, decision 81. The text owes a sentence stating the grid and that the quadrature was taken to convergence, and should note that the change lowers the KDE's absolute scores by 3 to 5 percent while leaving every out-of-sample comparison intact, because a reader will ask. |
| **Status** | RESOLVED in analysis; one sentence owed in the text. Decision 81. `audits/scoring_grid_error.py`. |

## 62. Post-stratification, and the empirical stratum shares were recorded for a 149-dataset arm

| | |
|---|---|
| **What** | The corpus allocates 2,500 datasets to each of four size bands for equal precision; the empirical arm is 13.9 / 54.2 / 26.4 / 5.6 percent. Since the KDE improves with n and the lognormal degrades, an unweighted corpus mean is a statement about the allocation as much as about the methods. `coverage.post_stratified` had existed since Stage 2a and **no stage had applied it to the W1 or rank results**. |
| **Effect** | It changes the sign of the corpus's family verdict on the mean: `Lognormal, Uniform` 0.1306 equal allocation against `KDE, Uniform` 0.1228, becoming 0.1240 against 0.1268 reweighted. On mean RANK the KDE stays ahead, 2.25 against 2.75 reweighted. Every headline aggregate is now reported both ways. |
| **And a stale constant** | `genconfig.EMPIRICAL_STRATUM_SHARE` was measured on the 149-dataset arm, before decision 61 dropped `Chairs` and `Grouting`. Corrected to 20 / 78 / 38 / 8 over 147. It moves `coverage.post_stratified`'s medians in notebook 1 by at most 0.27 percent relative, the largest being `weight_outliers` 0.036651 to 0.036550. `recovery.empirical_size_shares` now MEASURES the shares from the arm it is given, so this class of staleness cannot recur in the score tables. |
| **Note for the text** | Three empirical datasets exceed the corpus maximum of n = 9,999 -- the three largest `ReadyMix` strength classes, at 14,366, 20,814 and 31,025 -- so the reweighting covers 144 of the 147 and renormalizes over the four bands both arms share. |
| **Fix** | **Text.** State that the corpus is equally allocated by design and that headline aggregates are reported reweighted to the empirical size mix as well. |
| **Status** | Open. `TABLE_PostStratified.csv`. |

## 63. The corpus parent is not reconstructible from what the corpus stores

| | |
|---|---|
| **What CONTEXT.md said** | `parents.json.gz` holds "the parent of each dataset, enough to rebuild its CDF exactly". |
| **What is true** | It holds how each parent was ASKED for: each component's moment targets, from which `components.solve_component` recovers its location and scale deterministically, plus the global shift and the truncation bounds. It does NOT hold the displacement the overlap solve gave each component, which is one solved scalar times k ordinates drawn from the generator's stream. One recorded overlap value cannot identify k - 1 displacements. |
| **Fix** | **Code, done.** `corpus.rebuild_parents` replays the generation loop, which is deterministic given the seed, and keeps the parent objects `generate_corpus` discarded. It is not a regeneration: no corpus is written, nothing is redrawn, and the replay is checked rather than trusted -- it refuses unless `genconfig.DEFAULT` still equals the recorded configuration, compares twelve record fields plus `pi`, `market` and `mode_counts` per dataset, and compares the replayed values and weights against `values.parquet` element by element. All 10,050 datasets of `corpus_2026-09-15b` replay byte-identically. |
| **Status** | RESOLVED in code. **No manuscript consequence**; recorded because CONTEXT.md asserted something false about the deposit and a reader of the code would have believed it. |

## New, found during the Stage 2c author review

## 64. The variable-weighting penalty below n = 100 is the flat Dirichlet, not weighting

| | |
|---|---|
| **What entry 55 said** | On a common target, variable weighting is a coin flip overall and strongly size dependent: for the KDE, worse by 0.0395 at n = 3-9 and better by 0.0571 at n >= 1000. |
| **Why that is not the whole story** | A synthetic dataset's weights are built in two steps. Mode k is given its true market share, which is SIGNAL -- the market-weighted parent is a real population object at `mode_coupling = 1.0`. That share is then split among the points inside mode k by a FLAT DIRICHLET, which is noise the real world does not have, because a market share is a property of a product and not a random draw. So the measured penalty may be a property of the STAND-IN rather than of weighting as a practice. |
| **The counterfactual** | Refit everything under ORACLE weights: the same mode-level share, split EQUALLY within each mode. Same signal, no within-mode noise. Not a method anyone could use; it isolates the stand-in. |
| **Result, paired against the same uniform fit, against the market parent** | At **n = 10-99** the penalty essentially vanishes and stops being distinguishable from zero: KDE **-0.0398 to -0.0056**, lognormal **-0.0335 to -0.0014**, normal -0.0235 to -0.0066. That is 86, 96 and 72 percent of the penalty. At **n >= 100** the oracle makes variable weighting BETTER than the realized weights do: KDE +0.0274 to +0.0437 at n = 100-999 and +0.0513 to +0.0575 above 1,000. At **n = 3-9** a real penalty survives for the lognormal, -0.0213 and still distinguishable, because estimating a several-mode market mixture from three to nine points does not work however clean the weights are. |
| **Fix** | **Text, and it changes a claim.** Do not write "variable weighting hurts below n = 100". Write that variable weighting pays whenever the market shares are actually known, from about n = 10 upward, and that the penalty this study measures below n = 100 is the price of representing UNKNOWN shares with a flat Dirichlet. **This is the strongest argument in the project for the real production volumes of Marsh, Hattam and Allen (2025)**, and it should be said where that paper is cited. |
| **Status** | Open. Decision 73. `audits/weight_noise_vs_signal.py`, `TABLE_WeightNoiseVsSignal.csv`. |

## 65. Why the KDE loses at n = 10-99, and the three explanations that are excluded

| | |
|---|---|
| **The objection** | A KDE is the most flexible method under test, so it losing to a three-parameter lognormal at n = 10-99 does not pass a sniff test. Asked four times across the project. |
| **Not the halving** | Cross-validation at fit fractions 0.5, 0.7, 0.8 and 0.9 gives an empirical deficit of -0.0670, -0.0684, -0.0663, -0.0665. It does not shrink as the fitting half grows; the 50/50 split is the protocol most favorable to the KDE of the four. |
| **Not the evaluation protocol at all** | Against the known parent, fitting on every value and splitting nothing, the KDE still loses at n = 10-99: -0.0172 uniform and -0.0228 variable, both distinguishable. |
| **Not the over-dispersion, which is the surprise** | A Gaussian KDE's variance is the data's PLUS h^2, and the fitted spread over the data's is 1.63 at n = 3-9 and 1.19 at n = 10-99 against the normal's 1.35 and 1.02. **Correcting it exactly does not recover the loss**: shrinking the points so the density regains the data's variance moves the n = 10-99 deficit only from -0.0138 to -0.0126 under uniform weighting, makes n = 3-9 worse under variable, and beats the plain KDE on 47 to 55 percent of datasets. |
| **Partly the guard, but only partly** | Under pure Silverman the same gap is -0.0085 instead of -0.0138 (uniform) and -0.0103 instead of -0.0177 (variable). The guard costs the KDE about 40 percent of the deficit and does not cause it. |
| **The mechanism** | The ordinary bias-variance tradeoff. A parametric family converges at root-n and a KDE at n^-2/5, so at small n the lognormal's shape bias costs less than the KDE's variance, and at large n the bias stops shrinking while the variance does not. **The crossover is at n of about 100, which is where it is observed.** With 30 points a bumpy nonparametric estimate of a smooth truth loses to a smooth three-parameter one however its variance is scaled. |
| **Fix** | **Text.** State the crossover and its mechanism as a positive finding rather than defending the KDE. It is the honest answer to "when should a practitioner use kernel density estimation", and the answer is "when the category has about a hundred EPDs or more", which is a usable rule. |
| **Status** | Open. Decision 74. `audits/cv_fit_fraction.py`, `audits/kde_variance_correction.py`. |

## 66. In-sample W1 rewards flexibility, but this study does not let the KDE exploit it

| | |
|---|---|
| **What the Stage 2c draft said** | That scoring against the training data is a defect because it rewards the most flexible method. |
| **Why that was too broad** | Rewarding flexibility is the point of the comparison. The defect is narrower and it is that in-sample W1 has a DEGENERATE optimum: a KDE with a vanishing bandwidth scores exactly zero on any dataset, so the criterion cannot separate a good flexible method from an arbitrarily flexible one. **This study never lets the KDE reach that optimum**, because the bandwidth is fixed by a rule and that rule was calibrated on held-out likelihood, not on W1. The in-sample score is therefore optimistic for the KDE but not degenerate. |
| **What IS unambiguously circular** | The second defect, which is not a matter of degree: scoring uniform-weighted models against the VARIABLE-weighted empirical CDF charges them a distance no estimation method can remove, 61.9 percent of the score on the empirical arm. Entry 55. |
| **Fix** | **Text.** When the paper motivates the out-of-sample criteria, motivate them on the weighting circularity and on the complexity penalty, not on "flexibility is rewarded". A reviewer who knows kernel methods will notice the difference. |
| **Status** | Open. |

## 67. The guard threshold, swept on both criteria

| | |
|---|---|
| **The question** | Entry 60 reported that the parent referee prefers pure Silverman to the guarded rule, which invites the response that the threshold should be adjusted rather than the guard abandoned. |
| **Swept** | `SILVERMAN_MIN_NEFF` over 0, 5, 10, 15, 20, 30, 50, 100, 200 and infinity. **The two criteria disagree.** Held-out likelihood peaks at 20 to 30 on both arms; W1 against the parent peaks at **5**, where mean W1 is 0.1355 against pure Silverman's 0.1367 and the current 30's 0.1400. **So the parent referee argues against this THRESHOLD, not against the guard**, and entry 60 was too broad. |
| **MOVED TO 20, reversing this entry's first recommendation** | It said keep 30, on the grounds that moving it to improve W1 would be tuning on the reported criterion. That is wrong: the reported criterion is IN-SAMPLE W1 and the parent score is an independent out-of-sample truth. |
| **Why 20 and not 10, which is what a reviewer asks** | Stepping the threshold down one value at a time and measuring what each step buys in parent accuracy per unit of held-out likelihood it costs, **every step from 200 down to 20 is free or better than free** -- 30 to 25 buys 0.65 percent for 0.29, and 25 to 22 and 22 to 20 cost nothing. **The step 20 to 18 is the first that costs more than it buys**, at a marginal ratio of 0.34, and every step below is also below 1. The held-out p05 agrees: flat at about -1.62 from 200 to 18, then -1.65 at 15, -1.72 at 10, -1.88 at 5. |
| **Fix** | **Analysis, DONE**, plus half a sentence of text: the threshold was swept on both criteria, they disagree, and 20 is the smallest value reachable by steps that each cost nothing on the criterion the guard protects. |
| **Status** | RESOLVED in analysis. Decision 80, superseding 75. `audits/guard_threshold_sweep.py`. |

## 68. The bandwidth documentation described a rule the code does not run

| | |
|---|---|
| **What two comments said** | That `silverman_guarded` uses **Scott** below `SILVERMAN_MIN_NEFF`, which was 30 at the time and is 20 from decision 80. |
| **What runs** | `0.9 * scale * n_eff ** -0.2` throughout, with only the SCALE guarded: the robust `min(sd, IQR/1.34)` at or above 30 effective observations, the plain standard deviation below it. Scott carries **1.06** where this carries **0.9**, so the fallback bandwidth is 18 percent smaller than "Scott" would be. Verified numerically. |
| **Where** | `customstats.weighted_bw`'s main docstring had it right; its `min_neff` parameter note and `fitting.BW_METHOD`'s comment block did not. |
| **Why it matters** | A Methods paragraph written from either of those two would describe a different estimator from the one that produced every number in the paper. |
| **Fix** | **Code comments, done.** Nothing numeric moved: only prose was wrong. Decision 77. |
| **Status** | RESOLVED. |

## 69. The test the author actually wants, which no stage has run

| | |
|---|---|
| **The framing** | From the Stage 2c review: "the test should be, if we use this probabilistic model in the context of a probabilistic whole-building LCA, how faithfully do those probabilistic models represent the true population of data? Does it even matter?" |
| **Why it is the right test** | Every fit-quality criterion in this study, old or new, is instrumental. W1, overlap area, held-out likelihood and the recovery score all matter only insofar as they change a pLCA answer, and none of them has been shown to. |
| **Why it is newly possible** | Until Stage 2c the synthetic parents could not be reconstructed, so there was no way to run a pLCA on the TRUTH. `corpus.load_parent_objects` now returns an object exposing `ppf` and `rvs_from_uniform`, which is everything notebook 3 needs from a model. |
| **The experiment** | Run the pLCA twice on the same common random numbers, once with each method's fitted models and once with the true parents, and report how far each method's ECI Rank #1 Frequency is from the truth. |
| **Owner** | **Stage 2e**, which owns the pLCA construction and the common random numbers, and **2g**, which owns the metrics. Not run in 2c, which was told not to touch the pLCA construction. |
| **Fix** | **Analysis, and it may change the paper's conclusion in either direction.** If the methods' pLCA answers are indistinguishable from the truth and from each other, that is the cleanest result the paper could report and it reframes the whole comparison. |
| **Status** | Open. Decision 65 gives the machinery. |

## 70. "95 percent of ECC datasets have one visible mode" is a statement about Scott's bandwidth

| | |
|---|---|
| **Manuscript and CONTEXT.md** | Quote roughly 95 percent of empirical ECC datasets as having exactly one VISIBLE mode, as a property of real ECC data, and use it to argue that real categories are single right-skewed humps with shoulders rather than separated humps. |
| **Code** | `modality.n_modes_visible` counts local maxima of `scipy.stats.gaussian_kde(x)` at its DEFAULT bandwidth, which is Scott's rule. Stage 2c established independently, on W1 against the known parent, that **Scott oversmooths this data by about 35 percent**. |
| **Numbers** | Share with exactly one visible mode, n >= 8, as the bandwidth is scaled from scipy's default: 1.20 -> 96.9 pct; **1.00 -> 94.6 pct**; 0.90 -> 91.5; 0.80 -> 80.8; **0.74, the Scott correction -> 73.1**; 0.60 -> 55.4; 0.50 -> 41.5. |
| **The part that matters more** | **The corpus-to-arm AGREEMENT is also a property of the bandwidth, and decision 38 tuned the generator against it.** Total variation between the arms is 0.0123 at scipy's default -- the figure the corpus was matched on -- and 0.0714 at the corrected bandwidth. The gap is datasets with three or more visible modes: **8.5 percent of the empirical arm against 1.3 percent of the corpus.** |
| **Which way it cuts** | **Against the KDE.** Multimodality is the one structure a kernel estimate represents and a three-parameter family cannot, and the corpus has six times fewer strongly multimodal datasets than the arm. Any worry that the corpus was built to flatter the KDE is the opposite of what this shows. |
| **Fix** | **Text at minimum, and an author decision beyond that.** The reported figure must be restated with its bandwidth, quoting the table, because a reviewer who recomputes it at any other smoothing gets a different answer. Whether to correct the measure and retune is decision 78; the recommendation is not to, because the gap understates the case for the paper's own method and a limitation stated against yourself is safe ground. |
| **Status** | Open. Decision 78. `audits/visible_modes_bandwidth.py`, `TABLE_VisibleModesByBandwidth.csv`. |

## 71. Which categories the KDE wins on, and why the average the paper takes is the wrong one

| | |
|---|---|
| **What every aggregate does** | Weights each of the 147 categories equally. |
| **What the arm looks like** | 49 datasets reach n = 100, which is 33 percent of CATEGORIES and **97.3 percent of the EPDs**. 88.1 percent of all EPDs sit in datasets with n >= 1,000, the band where the KDE beats the lognormal on 76 to 99 percent of datasets. The concrete family -- ReadyMix, Shotcrete, CMU, Precast -- is 25 datasets and **77.1 percent of the arm by EPD count**, with every ReadyMix strength class between 3,974 and 31,025 values, and it is the largest single embodied-carbon contributor in most buildings. |
| **So** | The categories where the KDE wins decisively are the structural materials that dominate a building's embodied carbon; the categories where it loses are small and specialized. A category-count average says the lognormal and an average reflecting what a building is made of says the KDE. **Neither is currently reported.** |
| **Fix** | **Text.** Keep the category-count average as the primary, because it is the honest unweighted answer, and add a paragraph with the observation above. Do NOT invent an ad-hoc importance weight; the principled versions are Stage 2i's real-building anchor and entry 69's pLCA-against-truth. |
| **Status** | Open. |

## 72. The 2-parameter lognormal is the wrong comparator and the paper should say so

| | |
|---|---|
| **Why** | Against the known parent it is the worst of the four right-skewed families: +0.0323 behind the three-parameter form under uniform weighting and +0.0253 under variable, losing on 66.5 and 61.1 percent of datasets. Even the Stage 1 method the manuscript currently describes -- a lognormal with a fixed +0.5 offset -- is much closer to the three-parameter fit than the two-parameter one is. |
| **And the counterpart** | On REAL data the three-parameter lognormal is indistinguishable from gamma and from the two-parameter form out of sample (entry 59). The separation between them is a synthetic-arm result. |
| **Fix** | **Text.** State which lognormal is being compared, every time. "Lognormal" without a parameter count is ambiguous across a factor that matters more than the gap to the KDE at small n. |
| **Status** | Open. |

## 73. The method comparison by material, and the result the paper should lead with

| | |
|---|---|
| **Why** | Entry 71 says every aggregate weights the 147 categories equally and that this is not the question the paper asks. This is the stratification that answers it. `src/materialclass.py` splits the arm into the structural frame and its binders, the envelope, and everything else, from published building-LCA hot-spot practice. **It reads only the category NAME**, and `tests/test_materialclass.py` drives the whole assignment on a frame with no value column, so a tier cannot have been drawn after seeing which method won on it -- the same constraint decisions 43, 46 and 60 impose on the category rules. |
| **Result, empirical arm, cross-validated, share of datasets on which each method is closest within its weighting scheme** | **structure** (41 datasets, 98,216 values): `KDE, Uniform` **0.488** against `Lognormal, Uniform` 0.341; under variable weighting the lognormal leads 0.488 to 0.317. **envelope** (32, 3,869): the lognormal leads 0.625 to 0.188 and 0.719 to 0.062. **other** (54, 14,578): the lognormal leads 0.537 to 0.259 and 0.537 to 0.204. |
| **And the conjunction the paper should lead with** | **structural categories with n >= 100: 23 datasets holding 97,438 values, 83 percent of everything in the arm.** `KDE, Uniform` is closest on **69.6 percent** of them against the lognormal's 26.1, mean cross-validated W1 **0.0658 against 0.0737**; under variable weighting 47.8 against 43.5, 0.0808 against 0.0844. |
| **It is not a fished subgroup** | The tiers were fixed in `materialclass.py` before any result was looked at, and n = 100 is the crossover established independently in entry 65. It is a conjunction of two prior findings. |
| **What it is not** | An importance weight. It is a stratification, and every number in it is a plain mean within a named group. Weighting categories by n was considered and rejected: that weights by how many EPDs a manufacturer happened to publish, which correlates with the dimension the KDE wins on. The principled versions are Stage 2i's real-building anchor and entry 69's pLCA-against-truth. |
| **Fix** | **Text, and it is the paper's strongest honest claim.** Report the unweighted category average as primary, then this. `TABLE_MaterialTiers.csv` must be published so a reader can check the classification. |
| **Status** | Open. Decision 83. `CompareUQMethods_FIG_MethodByMaterial.png`. |

## 74. Notebook 2 held a second copy of the scoring criterion

| | |
|---|---|
| **What** | Notebook 2 cell 23 computed the synthetic arm's W1 with an inline `wasserstein1_weighted` call instead of `fitting.score_w1_model`. It was a second implementation of the study's criterion, in the notebook, outside the tested module. |
| **What it cost** | When `fitting.W1_ROUTE` moved to trapezoid quadrature (decision 81) the cell silently kept the old one, so the same quantity appeared in `TABLE_SyntheticECCMetricsAndW1.xlsx` and in `TABLE_MethodScores.csv` with values differing by **up to 9 percent**. |
| **How it was found** | `tests/test_regression.py::test_synthetic_fits_and_w1_recomputed`, which drives the production path and compares it to the notebook's own output. That is precisely what the test exists for and it worked. |
| **Fix** | **Code, done.** Cell 23 calls `score_w1_model`. Notebook 3's two inline `wasserstein1_weighted` calls are a DIFFERENT quantity -- W1 between two fitted models, with no empirical CDF in it -- and are correct as they stand. |
| **Why it is in this log** | It is the same duplication Stage 1 removed from the FITTING block, reappearing in the SCORING block, and it says the lesson has to be applied to every criterion the paper reports, not only to the fit. **No manuscript number was ever published from the wrong copy**: it was caught in the same session that created it. |
| **Status** | RESOLVED. |

## 75. The material tier adds nothing beyond dataset size

| | |
|---|---|
| **What entry 73 claimed** | That the KDE is closest on 70 percent of structural categories with at least 100 EPDs, presented as a material finding. |
| **What is actually true** | Size is the mechanism. Regressing `log(W1_KDE / W1_lognormal)` on `log(n)` and then adding the material tier as a factor, the tier adds nothing detectable: R2 0.316 to 0.321 under uniform weighting and 0.216 to 0.225 under variable, **F = 0.49 and 0.72, p = 0.61 and 0.49**. Within one size band the tier ordering is not stable. |
| **Why the tier looked like a mechanism** | Structural categories are the well-populated ones: **median n of 140 against 52 for envelope and 46 for everything else**, and the six largest categories in the arm are all ReadyMix strength classes. |
| **And what makes them different, which answers "are the others just more lognormal"** | **No -- they are better behaved in every way.** Median coefficient of variation 0.307 for structure against 0.828 and 0.736; skewness 0.917 against 1.566 and 1.716; excess kurtosis 1.412 against 3.716 and 5.312; and a HIGHER Shapiro statistic against BOTH the normal (0.911 against 0.828 and 0.791) and the lognormal (0.964 against 0.951 and 0.931). Concrete and steel are tight, nearly symmetric populations with many EPDs. Finishes and furnishings are sparse, dispersed and heavy tailed, and a two- or three-parameter skewed family is a good description of those. |
| **Fix** | **Text.** State ONE mechanism with a threshold: the KDE overtakes the lognormal at about **124 EPDs** under uniform weighting and 204 under variable, out of sample on the empirical arm, and the materials that dominate embodied carbon are the ones that clear it. Do NOT report "structural and n >= 100" as a separate finding; it overstates what the data supports. The figure plots the log ratio against n colored by tier. |
| **Status** | Open. Decision 84, narrowing 83. `TABLE_SizeVersusMaterial.csv`, `TABLE_CharacteristicsByTier.csv`. |

## 76. The scoring grid did not charge the lognormal for its tail

| | |
|---|---|
| **What** | W1 was integrated on a linear grid to `max(x) + 10 sd`. Above that the empirical CDF is 1, so the integrand is the model's survival function, and everything the model puts out there was simply not counted. |
| **It is not symmetric across methods** | The truncated normal and the KDE put **exactly zero** mass above that point. The three-parameter lognormal puts a mean of 1.8e-4 and up to 4.9e-3. So the omission under-charged one family and not the others. |
| **Fix** | **Analysis, done.** `fitting.W1_TAIL_TERM` adds the mean excess above the grid on a log-spaced extension to the 1 - 1e-10 quantile. It raises the lognormal's mean W1 by 0.28 to 0.41 percent, leaves the other four unchanged to five decimal places, and takes the worst-case relative error against a +400 sd reference from 3.1e-2 to 1.6e-3. Extending the linear grid instead would need five times the points to hold the resolution that entry 61 established as the binding constraint. |
| **Note for the text** | This is the third criterion change in a row that moves numbers in the KDE's favor -- the bandwidth guard, the quadrature, and now the tail -- each for an independently correct reason. **Present them as one paragraph about taking the criterion to convergence**, with the convergence tables, rather than as three separate improvements, because three separate improvements all helping one method reads badly however sound each is. |
| **Status** | Open as a text item. Decision 85. |

## 77. Which method a practitioner should default to, answered against an oracle

| | |
|---|---|
| **Why this and not the method comparison** | Every other table compares methods. A practitioner must pick one RULE and apply it to every material in a building, so the decision-relevant comparison is between POLICIES, measured against the unreachable oracle that picks the best method for each dataset. |
| **The assumption being tested** | That kernel density estimation is the safe default: flexible enough that you never have to decide, and trustworthy everywhere. **It is half right.** |
| **Results, uniform weighting, mean cost over the oracle and worst single dataset** | always normal 71.2 pct / 5.38x empirical, 98.1 pct / 58.1x synthetic. always lognormal **4.9 pct** / 2.75x, 23.0 pct / 24.5x. always KDE 15.3 pct / **1.61x**, **14.9 pct** / **6.88x**. **KDE if n >= 100 else lognormal: 4.5 pct / 1.57x empirical, 9.4 pct / 6.79x synthetic.** |
| **What is true about the KDE** | **It never fails badly.** Worst case 1.61x against the lognormal's 2.75x on real data, 6.9x against 24.5x on the corpus. That is the defensible form of "safe to default to". |
| **What is not** | That it is the most accurate default. On real data out of sample, always-lognormal costs 4.9 percent over the oracle and always-KDE 15.3, and the KDE is within 5 percent of the best method on 47.2 percent of datasets against the lognormal's 66.9. |
| **Fix** | **Text, and it should be the paper's recommendation.** Use the KDE above about 100 EPDs and a parametric family below. It beats both fixed defaults on both arms and on both axes, and it is a rule a practitioner can apply without judgment. Report the KDE's worst-case advantage separately, because it is the honest version of the flexibility argument. |
| **Status** | Open. Decision 86. `TABLE_PolicyComparison.csv`, `CompareUQMethods_FIG_MethodByMaterial.png`. |

## 78. The study covers one impact category, and the text must say so

| | |
|---|---|
| **What** | Every ECC in this study is global warming potential in kgCO2e. The EPDs behind them also report acidification, eutrophication, smog formation, ozone depletion and primary energy, and none of those is examined. |
| **Why it matters for the conclusions** | The findings are all about the SHAPE of a distribution -- dispersion, skewness, modality, and how many values a category has. There is no reason to assume those are the same for acidification as for GWP, and the recommendation this paper makes is a threshold on dataset size that depends on them. A category with 400 GWP values may have far fewer for eutrophication, since reporting completeness varies by indicator. |
| **Fix** | **Text, one paragraph in the limitations.** State the scope plainly, state that the size threshold is the part most likely to transfer (it is a property of estimation, not of the indicator) and that the characteristic distributions are the part least likely to. Do NOT extend the study; the author's judgment that it is too large a rabbit hole is right, and a claim about one indicator honestly bounded is worth more than a rushed pass at five. |
| **Status** | Open. Raised by the author 2026-09-17. |

## 79. Multimodality does not belong in the method-selection rule

| | |
|---|---|
| **The expectation** | That multimodality drives the choice between a kernel estimate and a parametric family, since representing several humps is the thing a KDE can do and a three-parameter family cannot. The manuscript is written around modality as a central characteristic. |
| **Incremental predictive power over log(n)** | **`n_modes` is last of eleven characteristics on both weightings**: incremental R2 of 0.00002 and 0.00104, p = 0.95 and 0.69 on the empirical arm. On the synthetic arm, where 2,415 datasets make almost anything detectable, it reaches 0.004 and 0.009, against `log(n)`'s own R2 of 0.616 and 0.395. |
| **And as a selection rule it is actively harmful** | Cost over the oracle, empirical cross-validated, uniform weighting: `n >= 200` 3.56 percent, `n >= 100` 4.46, always lognormal 4.87, **`2+ modes only` 6.17**, `n >= 100 or 2+ modes` 7.06. Selecting on modality is worse than not selecting, and it triples the worst case from 1.575 to 2.755. |
| **Why the intuition fails** | The KDE's advantage is general shape matching -- skewness and tail behavior -- rather than resolving separate humps, and at the bandwidth the study fits 68 percent of empirical datasets are visibly unimodal. Entry 70 and section 4.10 already showed the advantage is present in the unimodal majority. |
| **What does predict the gap** | `w_v_uw_wasserstein`, the uniform-to-variable distance, is the strongest single addition and replicates on both weightings, incremental R2 0.046 and 0.055, p = 0.004 and 0.003. It is a property of the WEIGHT VECTOR and not of the data, a practitioner can only compute it after choosing weights, and it belongs to Stage 2d. Noted, not used. |
| **Fix** | **Text, and it demotes a characteristic the manuscript treats as central.** State that modality was tested as a selection criterion and rejected on both axes, and that the rule is one threshold on one number. Stage 2f owns the full metric reduction; this is a targeted answer. |
| **Status** | Open. Decision 88. |

## 80. How safe is assuming uniform weights? A per-dataset probability, for Stage 2d

| | |
|---|---|
| **The question** | The study samples market shares from a flat Dirichlet, so every draw is an allocation it considers possible. For a given dataset, what proportion of those possible allocations differ enough from uniform to matter? That turns the weighting question from a population average into something a practitioner can apply to the category in front of them. |
| **Why it is not answered here** | It needs a threshold for "enough to matter", and Stage 2d owns that: the named relative measure and the flip probability calibrated against it. A placeholder threshold would make the number unciteable. |
| **Feasibility, measured** | 147 datasets by 300 Dirichlet draws in **7 seconds**. `audits/weighting_risk.py`. |
| **What it looks like** | Against a placeholder of a tenth of the mean, the median probability is 0.847 at n = 3-9, 0.675 at 10-99, 0.218 at 100-999 and **0.000 above 1,000**. Uniform weighting is safe for 36 of 147 datasets and almost never safe for 12. `PaintingAndCoating` (n = 19, CV 1.33) is riskiest; `Asphalt` (n = 7,232, CV 0.27) is safest. |
| **THE FINDING THAT MAKES IT WORTH A PARAGRAPH** | **Dispersion beats size here, and nowhere else in this study.** Spearman with the coefficient of variation +0.693 and with the interquartile range +0.621, against log(n) at -0.569. Every question about which METHOD fits best is driven by n; whether WEIGHTING matters is driven by spread. Those are different mechanisms and the paper should say so, because a reader who has absorbed "it is all about n" will assume it applies here too. |
| **It also supersedes a characteristic** | `w_v_uw_wasserstein` is ONE draw from this distribution. Reporting the distribution retires the open item about a single Dirichlet realization moving per-dataset metrics a long way. |
| **The instrument to use, and it is not the probe's** | **A_IQR, from the author's own KL2 paper**: sample weight vectors from the Dirichlet, fit a PDF under each, and take the area of the interquartile range of that ensemble. Higher A_IQR means a wider error band around the uniform-weights PDF. It is one number per dataset, it lives in density space where a practitioner reads an error band, and it is already published, so this paper can cite rather than re-derive. `audits/weighting_risk.py` used a thresholded CDF distance instead; that was a feasibility probe and not a proposal. |
| **Settled by the author, 2026-09-17** | The quartiles are **pointwise in x** across the ensemble, and the component PDFs use the **guarded Silverman bandwidth** (`fitting.BW_METHOD`) to align with the rest of the study. The second may be a deliberate divergence from KL2, and the paper must say so if it is: this study's bandwidth is fixed by decisions 54 and 80, and a weighting-risk measure at any other bandwidth would not describe the densities this paper fits. |
| **Still to read off the paper** | How the area is normalized, if at all, and how many Dirichlet draws. `refs/1-s2.0-S0921344926002466-main.pdf`. A number sharing KL2's name but not its definition is worse than a new one. |
| **Fix** | **Analysis, Stage 2d**, then text. Cite KL2 for the measure and state any divergence from it explicitly, as decision 9 required for the bandwidth. |
| **Status** | Open, specified, owner 2d. Decisions 89 and 90. |

## 81. The ICE figure behind the plausibility ceiling is UNSOURCED, and the ceiling does not need it

| | |
|---|---|
| **What the record says** | `empirical.MASS_ECC_CEILING = 100.0` kgCO2e/kg is justified by two independent arguments. The first: published cradle-to-gate inventories put building-product coefficients at 0.1 to 15 kgCO2e/kg, the highest being primary aluminium, which the Inventory of Carbon and Energy (ICE) database v3.0 (Jones and Hammond, Circular Ecology, 2019) is said to place near 13 kgCO2e/kg. The second is stoichiometric and cites nothing. |
| **The problem** | The ICE figure came from a session's own knowledge. `refs/` holds no copy of ICE v3.0 and no other file in this repository contains it, so nothing here can confirm the number, the edition or the page. Decision 49 already flagged it: "verify it against the source before it goes in the paper." Stage 2d checked, and it cannot be verified from anything the repository holds. |
| **Resolution** | **Do not cite ICE in the manuscript.** The figure is recorded as unsourced rather than removed, because it is probably right and a later session with the database in hand can restore it with an edition and a page. Until then it is not citable. |
| **What the ceiling rests on instead, and it is sufficient alone** | Combusting pure carbon yields 44.009/12.011 = **3.664 kg CO2 per kg of carbon**. So 100 kgCO2e per kg of DELIVERED PRODUCT requires burning **27.3 kg of pure carbon for every kilogram shipped**, and even 25 kgCO2e/kg requires 6.8 kg. No inventory database is needed to see that a building product cannot carry 27 times its own mass in combusted carbon. The argument is arithmetic and a reviewer can check it in one line. |
| **Why this changes nothing about the ceiling** | The bound is external either way, which is the property decision 60 requires: the threshold is anchored outside the data rather than read off the arm's own spread. It also stays deliberately loose at 100 rather than 25, so it cannot be read as tuned, and it still catches the known cases by two orders of magnitude -- two `Elevators` records at 20,812 and 21,945 kgCO2e/kg, and 87 `Cement` records reporting a per-tonne GWP against a 1 kg declared unit. |
| **Fix** | **Text.** State the ceiling with the stoichiometric justification and no database citation. If ICE is wanted as corroboration, someone must open v3.0 and record the edition and page. |
| **Status** | RESOLVED as far as this repository can take it. The ICE citation is withdrawn; the ceiling stands on the arithmetic. Decision 49 amended in place. |

## New, found in Stage 2d

## 82. WITHDRAWN. The "5.33 percent noise floor" is not a change in any reported number

| | |
|---|---|
| **What this entry said** | That the study's pLCA compares UQ methods under independent random streams, that running one method twice changes the answer 5.33 percent of the time, and that this blocks reading a flip probability off the study's results. |
| **Why it is withdrawn** | The 5.33 percent counts how often the LABEL "which material has the highest rank-1 frequency" lands on a different material. It is not a change in any quantity the study reports. Worked case: the four frequencies were 0.2335, 0.2581, 0.2597, 0.2487 in one run and 0.2325, 0.2583, 0.2543, 0.2549 in another. **The largest change in any number is 0.0062.** The top two differed by 0.0016, inside their own 0.0043 standard error, so an arbitrary tie-break went the other way. 10,000 draws is ample. |
| **Why quoting it was misleading** | "5.33 percent of comparisons change" reads as though a material's contribution moved from 25 percent to 20 percent. Nothing of the sort happens, and no version of this number belongs in the manuscript -- nor does any quantity expressed as a multiple of it. |
| **What replaces it** | Entry 95, which measures what switching UQ method does in units a reader can act on. |
| **Status** | WITHDRAWN. Nothing here goes in the paper. |

## 83. The uniform-to-variable distance is mostly a SHIFT OF THE MEAN, which simplifies the practitioner rule

| | |
|---|---|
| **The question** | `w_v_uw_wasserstein`, the study's headline weighting characteristic, is W1 between the uniform-weighted and the variable-weighted version of one dataset. W1 is bounded below by the absolute difference in the two means, so the characteristic may be substantially measuring how far reweighting moves the mean rather than any change of shape. |
| **The split** | `location = abs(weighted mean - unweighted mean)`, which is that bound exactly; `shape = W1 - location`, non-negative by the inequality. `src/weighting.py`, table `TABLE_WeightingLocationShape.csv`. |
| **The answer** | **Mostly location.** Median location share **0.725 over the 147 real datasets and 0.804 over the 10,000 synthetic ones**, pooled 0.728 and 0.799, above half on 68.7 and 72.4 percent of datasets. On the real data it is highest where datasets are smallest: 0.96 at 3 to 9 EPDs, 0.65 at 10 to 99, 0.74 at 100 to 999 and 0.54 above 1,000. The inequality holds throughout; the worst residual over all 10,147 datasets is -7.0e-14, which is floating point in the quadrature. |
| **Why it is a good outcome** | A practitioner who wants to know whether market shares matter for their category does not need a distributional calculation. They need a weighted mean, which is a spreadsheet column, and the guidance can be stated that way. |
| **Fix** | **Text.** State the decomposition and the share, and state the practitioner rule in terms of the weighted mean rather than a Wasserstein distance. |
| **Status** | Open. Decision 92. |

## 84. A_IQR DOES NOT MEASURE WHAT DECISION 90 EXPECTED IT TO, and the reason is dimensional

| | |
|---|---|
| **What was expected** | Decision 90 adopted A_IQR from the author's KL2 paper as the instrument for "how safe is assuming uniform weights", on the reasoning that the Stage 2c probe found dispersion rather than size to be what drives whether weighting matters (Spearman +0.693 with the coefficient of variation against -0.569 with log n), and that A_IQR, being a dispersion-of-the-density measure, would inherit that. |
| **What is measured** | It does not. Over the 147 real categories, Spearman correlation of A_IQR with the coefficient of variation is **+0.042** and with log dataset size **-0.946**; over a stratified sample of 400 synthetic datasets, -0.277 and -0.993. |
| **The probe's finding survives on the other measure, and it is a BOTH-MATTER result rather than a reversal** | The median separation between the uniform-weighted fit and a drawn one, in units of the dataset mean, correlates with the coefficient of variation at **+0.731** and with log size at **-0.545**, which reproduces the Stage 2c probe's +0.693 and -0.569 almost exactly. **Do not quote the thresholded probability instead**: it is saturated, with 46 percent of real categories at exactly 1.000 and 74 percent above 0.99, and its apparent +0.803 against -0.106 is carried by the untied minority. **Where dispersion genuinely dominates is WITHIN a size band**, and there it is nearly deterministic: +0.940 at 3 to 9 EPDs, +0.888 at 10 to 99, +0.955 at 100 to 999 and +0.833 above 1,000. Among the 38 unsaturated categories the mirror holds, size -0.726 and dispersion +0.040. |
| **Why. AN EARLIER VERSION OF THIS ENTRY EXPLAINED IT WRONGLY AND THE EXPLANATION IS WITHDRAWN** | It said A_IQR cannot see dispersion because it is exactly invariant under rescaling the data. It is invariant, verified to ten decimal places over seven orders of magnitude, **but so is the mean-relative separation used instead**, so invariance cannot be the distinguishing property. What separates them is what each divides by: A_IQR measures the density's uncertainty against that curve's own height and width, so the data's spread cancels twice and only the weight sampling noise survives, which is a question of how many points there are; the separation is an x-axis distance divided by the mean alone, so the spread-to-mean ratio survives, and that ratio IS the coefficient of variation. Measured at n = 60 over a 27-fold change in the coefficient of variation: A_IQR moves 0.296 to 0.307 while `A_IQR * sqrt(n)` stays between 2.21 and 2.38, and the separation moves 0.030 to 0.664 while the separation divided by the coefficient of variation stays between 0.114 and 0.138. |
| **And A_IQR is DOMINATED by size rather than blind to spread** | Within each size band of the real arm its rank correlation with the coefficient of variation is -0.008, +0.128, +0.644 and +0.405: monotone but small. A five- to tenfold change in dispersion within a band moves it by a factor of 1.07 to 1.87, against a factor of 12 across the size range, which is why the marginal correlation is only +0.042. |
| **What A_IQR IS good for** | It is the right answer to KL2's question, which is how confident the uncertainty MODEL is, and it is published, so this paper can cite rather than re-derive. It is reported for exactly that. |
| **What answers this paper's question instead** | The distance between the uniform-weighted fit and the fit under a drawn market share, in units of the dataset's own mean. That is not scale free in the same sense, it does respond to dispersion, and it is the axis the flip probability is calibrated on, because a pLCA ranks materials by absolute contribution. |
| **AND WHAT THAT MEASURE SAYS IS NOT REASSURING** | The probability that a possible market-share allocation carries at least a 5 percent chance of changing the top contributor is **0.909 at equal allocation across size bands and 0.928 reweighted** to the real mix of category sizes. By band the means are 0.945 at 3 to 9 EPDs, 0.992 at 10 to 99, 0.919 at 100 to 999 and **0.310 above 1,000**. **Uniform weighting is defensible only for the largest categories**, which in this arm means a handful of concrete strength classes and asphalt. |
| **Fix** | **Text.** Report A_IQR with a citation to KL2 and say plainly that it is a property of dataset size; make the practitioner statement on the mean-relative measure. Decision 90 said that where this paper differs from KL2 it must say so, and this is such a place: the divergence is not in how A_IQR is computed but in what it is asked to do. |
| **Status** | Open. Decisions 90 (narrowed) and 94. |

## 85. What a given W1 actually costs: the calibration curve, and what it says about the six methods

| | |
|---|---|
| **What the manuscript lacks** | It reports W1 between a fitted model and a target, and asks the reader to accept that a smaller W1 is better without ever saying what a W1 of, say, 0.05 does to an answer. Nothing in the study connected the goodness-of-fit scale to a decision. |
| **What was built** | For every probabilistic LCA, pairs of fitted models at a controlled separation, run on common random numbers so the Monte Carlo floor is zero, recording whether the identity of the top-contributing material changed and whether the full ranking changed. Logistic regression on log distance, an isotonic fit beside it, and a bootstrap that resamples pLCA GROUPS rather than rows, because the comparisons inside a group share four datasets and one set of variates. |
| **Why the six UQ methods could not supply the curve on their own** | They never sit close enough together. Over 37,500 comparisons the smallest relative W1 between any two of the six is 0.00022, and the flip rate in the lowest 2 percent of separations is already **14.1 percent**. All three levels being asked about lie below the observed data, and an isotonic fit returns the same crossing for all three because its first block is above the top of them. So the calibration set adds pairs at separations running continuously to zero: the same kernel estimate under uniform weights and under weights moved a fraction of the way toward a Dirichlet draw. |
| **The curve** | The probability that the top-contributing material changes crosses **1 percent at a relative W1 of 0.0018** (interval 0.0013 to 0.0023), **5 percent at 0.011** (0.0091 to 0.0126) and **10 percent at 0.025** (0.0217 to 0.0277), with an isotonic fit giving 0.0026, 0.0129 and 0.0271. **Two significant figures and no more**: an independent run on a different random stream gave 0.0015, 0.0099 and 0.0233, all inside those intervals and all differing in the third figure. Decision 95. |
| **The check that the device is legitimate** | If the curve describes the DISTANCE rather than where the distance came from, then the six real method pairs -- which are different distribution FAMILIES -- fall on the curve fitted from weighting pairs, wherever the two overlap. They do in the two bands holding most of the pairs: at separations of 0.036 to 0.064 the calibration gives 0.176 and the method pairs 0.180 over 894 comparisons; at 0.064 to 0.124, 0.281 against 0.288 over 4,595. **They diverge at the top**, above 0.124, where the calibration gives 0.436 and the method pairs 0.606 over 29,943 comparisons. A cross-family difference of a given size is more consequential than a reweighting difference of the same size, presumably because the families differ in the tails that decide a ranking. The curve therefore UNDERSTATES the flip probability for large cross-family differences and should be read as a lower bound there. Below a separation of 0.036 there are fewer than 200 method pairs in total and the comparison says nothing either way. |
| **THE CAVEAT THAT HAS TO TRAVEL WITH EVERY ONE OF THESE NUMBERS** | Every material in this study is normalized to a mean of 1.0 and carries a material use intensity of 1.0, so the four contributions in a pLCA are nearly exchangeable and their ranking is as fragile as it can be made. A real building, where materials differ by orders of magnitude in contribution, is much harder to flip. These crossings are an upper bound on how often a modeling choice changes an answer, which is the conservative direction for a practitioner rule but must not be quoted as a statement about buildings. |
| **Fix** | **Text.** This is a new result and a new figure, `CompareUQMethods_FIG_FlipCalibration.png`. It is what turns the study's W1 scale into something a reader can act on. |
| **Status** | Open. Decisions 91, 93 and 95. |

## 86. The relative measure was already there, unnamed, and the normalization is not doing secret work

| | |
|---|---|
| **The situation** | Every dataset in this study is divided by its own unweighted mean before anything else happens, so every W1 the study has ever reported is already a W1 divided by a mean. The manuscript nowhere says so, and a reader cannot tell whether a reported 0.05 is an absolute distance in kgCO2e per declared unit or a relative one. |
| **What was done** | The measure is named and defined explicitly, and the claim is verified rather than asserted: the same quantity was recomputed on the RAW, un-normalized empirical values, in their own units, with dataset means spanning several orders of magnitude. The relative measure is unchanged to within floating point; the absolute W1 moves by exactly the rescaling factor, which is the control that the test is testing something. |
| **The two robust alternatives** | Dividing by the interquartile range or by the standard deviation instead. All three are computed with UNIFORM weights, which is the decision that matters here: a denominator taken under the variable weights would move when the weights move, which is the quantity being measured, and a practitioner holding a set of EPDs cannot compute a market-weighted mean without already knowing the market shares. |
| **Which to use** | The mean. See decision 93 for the measured comparison. |
| **Fix** | **Text.** State that scores are relative to the dataset mean, give the definition once, and say that the alternatives were tested. It moves no number: on a dataset normalized to a mean of 1.0 the named measure IS the reported W1, which `tests/test_weighting.py` pins. |
| **Status** | Open. Decision 93. |

## 87. A smoke run was committed, and the guard that exists for it is a sentence rather than a check

| | |
|---|---|
| **What happened** | Stage 2d ran notebook 3 under `COMPAREUQ_SMOKE_COMBOS=40` to exercise its new cells, then committed with `git add -A` while that output was still on disk. Commit `b750673` therefore replaced `outputs/tables/TABLE_PLCAResults.csv` with a 960-row smoke table in place of the 60,000-row result, **and overwrote seven figures with versions drawn from 40 probabilistic LCAs instead of 2,500**. The table was restored from the parent commit within the session; **the figures were missed by that restoration and only came back when the full notebook 3 run regenerated them**, which is itself the point below. |
| **What it cost** | Nothing, and the check that says so is strong. The regenerated table is BYTE IDENTICAL to the pre-smoke version and all seven regenerated figures are PIXEL IDENTICAL to theirs, so notebook 3 reproduces its entire output from its seed. No number was read while any of it was wrong, and the full test suite including the eight regression fixtures passed throughout. |
| **The part that is easy to miss** | Restoring the obvious artifact was not enough. The smoke run touched nine files and the fix addressed one, because that was the one whose damage was visible as a row count. A figure drawn from 40 groups instead of 2,500 looks like a figure. |
| **Why it is worth an entry anyway** | `CONTEXT.md` already says "Smoke results must never be committed", and that sentence did not stop it, because the smoke run and the commit were separated by half an hour of unrelated work. A rule that depends on remembering what a previous command did is the weakest kind. |
| **The cheap fix** | Notebook 3 already knows it is in smoke mode. It should redirect every write -- tables AND figures -- to a scratch directory when `COMPAREUQ_SMOKE_COMBOS` is set, rather than writing to `outputs/` at all. A weaker version is to have the run metadata carry the group count and a test assert it equals 2,500, but that only protects the tables, and this incident shows the figures are the part that gets forgotten. |
| **Fix** | **Code, owner Stage 3**, which owns the output conventions. Not done here, because it touches every table notebook 3 writes and Stage 2d had no mandate for it. |
| **Status** | **RESOLVED in Stage 2e**, which reruns the artifact this protects and so did it rather than leaving it to Stage 3. Every path notebook 3 writes goes through a redirectable output root; see entry 102. |

## 88. When does weighting matter? A closed form in two numbers a practitioner already has

| | |
|---|---|
| **The gap** | The manuscript treats the uniform-to-variable distance as a characteristic to be reported, not as something a reader can predict for their own category. A practitioner holding a set of EPDs has no way to ask "does this apply to me". |
| **What was measured** | The log of the separation between the uniform-weighted fit and a Dirichlet-weighted one, regressed on log dataset size and log coefficient of variation. **Size alone explains 48.0 percent of the variance, dispersion alone 49.5 percent, and both together 99.1 percent**, each adding about half on top of the other. They are nearly orthogonal, which is why neither alone looked like the answer. |
| **The fit** | `log(separation) = -0.318 - 0.434 log(n) + 1.036 log(CV)` on the empirical arm, R2 = 0.991; `-0.405 - 0.427 log(n) + 0.977 log(CV)` on 400 synthetic datasets, R2 = 0.996. **The exponents agree across the two arms**, which is what makes it worth stating as a law rather than as a fit. Practitioner form: **separation is about 0.73 * CV * n^-0.43**. |
| **The rule that falls out** | Combined with the calibrated 5 percent flip threshold of 0.011, uniform weighting is safe only when the coefficient of variation is below about **0.015 * n^0.43**: 0.046 at 10 EPDs, 0.120 at 100, 0.315 at 1,000, 0.826 at 10,000. The median real category is CV 0.63 at 47 EPDs and does not clear it. |
| **Why this is the strongest practitioner-facing result in the study** | It needs no distributional machinery, no Dirichlet sampling and no kernel estimate. A reader counts their EPDs, computes a coefficient of variation in a spreadsheet, and gets an answer. Every other rule this project has produced needs the analysis to have been run. |
| **The contrast the text must draw** | Every question in this study about WHICH METHOD fits best is driven by dataset size and nothing else, and modality in particular was tested and rejected. Whether WEIGHTING matters is driven by size AND dispersion. A reader who has absorbed the first will carry it into the second and be half wrong. |
| **Fix** | **Text**, and it is new. `TABLE_WeightingRiskDecomposition.csv`, `TABLE_WeightingRule.csv`, figure `CompareUQMethods_FIG_WeightingDrivers.png`. |
| **Status** | Open. Decision 96. |

## 89. THE FLAT DIRICHLET UNDERSTATES THE WEIGHTING RISK, so every number from it is a lower bound

| | |
|---|---|
| **The author's question, 2026-09-17** | A flat Dirichlet explores the simplex uniformly, but real market share probably arrives in clusters, with a few related products carrying most of the volume. Is uniform exploration the right model? And if share does cluster, is that not almost a dataset with fewer points, so that the effective sample size already captures it? |
| **The test** | Three weight schemes compared at MATCHED Kish effective sample size, which is what makes this a test of that reading rather than of concentration. `flat`: a Dirichlet over all n points. `scatter`: share concentrated into k groups placed on randomly chosen products. `blocks`: the same k groups placed on contiguous runs of the SORTED values, so products with similar coefficients share their volume. 97 categories, 30 draws each. |
| **The answer, and it splits in two** | **Concentration behaves exactly as the author expected.** `scatter` gives 0.90 to 0.99 times the flat separation across effective-size bands, which is no difference. So the effective sample size does capture "fewer data points". **Coherence does not.** `blocks` gives **1.5 to 3.1 times** the separation at the same effective sample size, and the ratio grows with effective size. |
| **Why** | A contiguous block shifts the whole distribution one way, and that lands in the LOCATION term which entry 83 shows carries a median of 72.5 percent of the uniform-to-variable distance. Random concentration moves mass in directions that partly cancel. Concentration and coherence are different things and only the first is a sample-size effect. |
| **What it means for every weighting number in this paper** | They are computed under a flat Dirichlet, so they are **lower bounds**. If real market shares cluster by product similarity -- and the 63.75 percent share Marsh, Hattam and Allen (2025) report for one steel route says they do -- the true separations are larger and uniform weighting is even less safe than reported. **This is the conservative direction for the paper's conclusion**, which is what makes it publishable as a stated limitation rather than a hole. |
| **What would settle it** | Real production volumes, which is what the Marsh data provides and what this study lacks. Short of that, Stage 2h's concentration sweep should vary the BLOCK STRUCTURE and not only the Dirichlet concentration parameter, because the two are not the same knob. |
| **Fix** | **Text**, as a stated limitation with its direction and its measured size. `TABLE_WeightingClustering.csv.gz`, `TABLE_WeightingClusteringSummary.csv`. |
| **Status** | Open. Decision 97. |

## 90. Silverman's constant: 1.34 or 1.35, and neither is wrong

| | |
|---|---|
| **The discrepancy** | This study divides the interquartile range by **1.34** to estimate a scale; the author's KL2 paper divides by **1.35**. A reader comparing the two papers will notice. |
| **Which is correct** | Both are roundings of the same exact quantity. The interquartile range of a standard normal is `2 * 0.674490 = 1.348980`, so the unbiased divisor is **1.3490**. Dividing by 1.35 estimates sigma with a bias of **-0.08 percent**; dividing by 1.34, **+0.67 percent**. **1.35 is the closer rounding.** 1.34 is what Silverman's 1986 book prints and what most software carries, which is why it is the more common convention. |
| **What it does here** | The two differ by 0.75 percent in the bandwidth. Nothing in this study turns on it: the guard threshold was calibrated over a range of effective sample sizes from 5 to 200 and the bandwidth rule was compared against Scott, which differs by 18 percent in the coefficient. |
| **Fix** | **Text, one sentence.** State that this study uses 1.34, that the exact value is 1.349, and that KL2's 1.35 is the same rule to within 0.75 percent of the bandwidth. Do not change the code: 1.34 is the published convention and moving it would shift every KDE number for a 0.75 percent correction nobody asked for. |
| **Status** | Open, one sentence. Decision 98. |

## 91. The flip threshold is conditional on FOUR materials, and Stage 2e changes that

| | |
|---|---|
| **The author's question** | Part of the plan was to stop building every probabilistic LCA from exactly four materials. Why is the calibration still on groups of four, and should the sweep not come first? |
| **The answer, and the concern is legitimate** | The sweep over 2 to 12 materials per pLCA belongs to Stage 2e and has not run. **The calibrated crossings of 0.0018, 0.011 and 0.025 are therefore conditional on four materials**, and they will move when that sweep lands. The direction is predictable: with more materials competing there are more chances for a near-tie, so the flip probability at a given model distance should RISE and the thresholds should FALL. |
| **Why calibrating first was still the right order** | The threshold is needed to state any per-dataset weighting risk at all, and the machinery -- common random numbers, the model-to-model distance, the tempered calibration set, the cluster bootstrap -- is independent of how many materials a group holds. Stage 2e re-runs a calibration it does not have to design. |
| **What 2e must do** | Re-run `flip.weighting_calibration` at each group size in the sweep and report the crossings as a function of it. If they move materially, every weighting-risk probability in notebook 1 is recomputed at the four-material value's replacement. |
| **Fix** | **Analysis, Stage 2e.** Until then the paper must state the crossings as conditional on four materials of equal material use intensity. |
| **Status** | Open, owner 2e. Decision 99. |

## 92. Could an industry-average EPD stand in for the weighted mean?

| | |
|---|---|
| **The author's idea** | Since the uniform-to-variable distance is mostly a shift of the mean, a practitioner needs a weighted mean rather than a distribution. An industry-average EPD is in principle built from a more complete dataset than any individual product declaration, so could it serve as that weighted mean? |
| **Why it is a good idea** | It is, in principle, exactly the missing quantity: a production-weighted average over a population the practitioner cannot otherwise see. **It is also what the author's own KL2 paper already does under another name** -- KL2 uses industry-average ECCs as its target expected value, `EVtarget`, and locates a phantom kernel so the model reproduces it. So this paper would be reaching for an instrument its companion paper has already defined, which is the consistency this project requires. |
| **What has to be checked before it is used** | Three things, none of them settled here. Whether the industry average is production-weighted at all, or a simple mean over participating manufacturers, which is a different object. What population it covers, since a regional average cannot stand in for a global one. And whether its scope, system boundary and reference year match the product declarations it would be compared against. |
| **How it would be used** | Not as a replacement for the dataset, but as a CHECK on it: the distance between the unweighted mean of the EPDs a practitioner holds and the industry-average value is a direct, computable estimate of the location term, whose median share this stage measured at 0.725. That is a rule needing no Dirichlet sampling at all. |
| **Fix** | **Analysis, a later stage, and it is not currently owned.** Recorded here so it is not lost. It would need industry-average ECCs for the categories in the arm, which EC3 carries for some and not others. |
| **Status** | Open, unowned, worth doing. Decision 100. |

## 93. Every material carries a material use intensity of 1.0, which makes the ranking as fragile as possible

| | |
|---|---|
| **The author's question** | Should this assumption be challenged rather than accepted? |
| **What it does** | Every dataset is normalized to a mean of 1.0 and every material use intensity is 1.0, so all four materials in a probabilistic LCA contribute the same expected amount. Their ranking is then decided entirely by the tails, which makes it as unstable as it can be made. This is why the Monte Carlo noise floor is 5.33 percent, why the flip probabilities are high, and why the calibrated thresholds are small. |
| **Which direction it biases** | **Conservative for every claim this study makes.** A real building has materials differing by orders of magnitude in contribution, where the largest contributor is usually obvious and no modeling choice will dislodge it. So the reported flip probabilities are an upper bound on how often a method choice changes a real answer. |
| **What is already scheduled** | Stage 2e owns a dominant-material-use-intensity variant, which is exactly this challenge. Stage 2i owns an optional real-building anchor with realistic intensities. Neither has run. |
| **What the paper must not do** | Quote a flip probability as though it described a building. Every one of them is conditional on four exchangeable materials, and the text must say so in the same paragraph, not in a footnote. |
| **Fix** | **Analysis, Stage 2e then optionally 2i**, and text in the meantime. |
| **Status** | Open, owner 2e. Decision 101. |

## 94. The headline metric should be a SHIFT, not a FLIP

| | |
|---|---|
| **What the manuscript does** | Reports "ECI Rank #1 Frequency" and compares UQ methods by how often the identity of the largest contributor changes. |
| **Why that is the wrong headline** | The identity of an argmax is a discontinuous function of four nearly equal quantities, so it inherits every source of instability at once. With the same fitted models and two independent Monte Carlo streams it changes 5.3 percent of the time for rank-1 frequency, 12.3 percent for contribution share and 3.0 percent for variance importance -- with no model difference at all. The underlying continuous values are stable to about 7 percent of their spread. |
| **What a probabilistic LCA is actually for** | Magnitude and likelihood: which material contributes most, and how confident that is -- because those drive design decisions and data-collection priorities. `eci_rank_1` per material already answers both at once. The error was collapsing it to "which material has the highest one". |
| **The headline this supports instead** | **Which UQ method you choose changes a material's rank-1 frequency by a median of 0.048 for the closest pair of the six and 0.188 for the furthest** -- that is, your stated probability that a given material is the largest contributor moves by 5 to 19 percentage points. Stable, continuous, no noise floor, and far less dependent on the four-material construction. |
| **What else belongs in the set** | Variance importance, the share of total variance a material's uncertainty accounts for, which is the measure that tells a practitioner where to spend data-collection effort. It is already computed as `ui` in the pLCA table and nothing reports it. |
| **Fix** | **Analysis, Stage 2g**, which owns the metric set. Report the continuous shift as primary, variance importance alongside, and any argmax statistic with its noise floor beside it. |
| **Status** | Open, owner 2g. Decision 102. |

## 95. What switching UQ method actually does to a pLCA result, in units a reader can act on

| | |
|---|---|
| **What the manuscript lacks** | It compares UQ methods by goodness of fit and by how often the largest contributor changes. Neither tells a reader what is at stake: the first is a distance with no interpretation, the second is a discontinuous label on four nearly equal quantities. |
| **What was measured** | For every pLCA output, the change caused by switching UQ method, over 250 probabilistic LCAs and all fifteen pairs of the six methods, with both methods given the same Monte Carlo draws so the comparison is like-for-like. Each figure is the change for the MOST-AFFECTED of the four materials, which is the one a practitioner is deciding about. Every material contributes a mean of 1.00, so the numbers read directly. |
| **The result** | Estimated contribution **0.181** (90th percentile 0.519); 95th percentile of the contribution **0.413** (1.290); standard deviation **0.173** (0.467); coefficient of variation **0.175** (0.389); chance of being the largest contributor **0.126** (0.278); contribution to total variance **0.089** (0.276); share of the building total **0.030** (0.085). **Switching UQ method changes a material's estimated contribution by about 18 percent.** |
| **Two findings inside it** | **The spread outputs move most** -- the 95th percentile by 0.41 against the mean's 0.18 -- which is the right way round, because representing spread is what a UQ method is for, so that is where two of them should differ. And **the contribution to total variance moves least**, which makes "where should I spend effort collecting better data" the steadiest answer a probabilistic LCA gives, more robust than any magnitude it reports. |
| **What the study should report and does not** | The variance contribution. It is the output a practitioner would act on most directly and the one least sensitive to the modeling choice this paper is about. **CORRECTED in Stage 2g: the clause "appears in no table, no figure and no section of the manuscript" IS FALSE.** The manuscript reports it in Figure 5b and 5d, defines it in Supplement 3(c), and concludes from it that the methods give similar uncertainty indices. It is under-reported, not unreported, and the instruction is to promote it. See entry 145. |
| **Fix** | **Text, and it replaces the current framing.** Report these absolute changes rather than a goodness-of-fit ranking, and add the variance contribution to the reported set. Stage 2g owns the metric set. `TABLE_OutputMetricSensitivity.csv`. |
| **Status** | Open, owner 2g. Decisions 102 and 103. |

## New, found in Stage 2e

## 96. The pLCA compared UQ methods under independent randomness, and now does not. What that was worth

| | |
|---|---|
| **What the code did** | Notebook 3's pLCA loop drew each UQ method's Monte Carlo sample from its own stretch of one shared stream, so two methods were compared under two independent sets of random numbers and any difference between them mixed the difference between the models with the difference between the draws. |
| **What it does now** | One uniform variate per material per Monte Carlo iteration, drawn once for the group and pushed through every method's inverse CDF: independent across materials within an iteration, identical across methods. The capped-reduction strategy is paired too. This is the practice Henriksson et al. (2015) and Heijungs (2021) recommend for comparative probabilistic LCA and that Marsh et al. (in press) use, so the manuscript can now cite all three for it. |
| **What it was worth, measured** | Over 300 probabilistic LCAs and all fifteen method pairs, with each figure the change for the most-affected of the four materials and every material contributing a mean of 1.00: the estimated contribution moves **0.1819** between two methods on shared draws and **0.1828** on independent ones, while running ONE method twice moves it **0.0095**. The sampling noise is **4 to 15 percent** of the model difference across the ten outputs and adds nearly orthogonally to it, so the study's old measurement was not inflated -- every ratio of unpaired to paired lies between 0.99 and 1.02. |
| **Where it does matter** | The argmax. The top contributor changes in **3.67 percent** of comparisons with NO model difference at all, which is a floor under any flip statistic read off the old table, and common random numbers take it to exactly zero. That is the honest, measured version of the claim entry 82 was withdrawn for overstating. |
| **What moved** | Every row of `TABLE_PLCAResults.csv`, by Monte Carlo noise, and no aggregate. Per row `eci_rank_1` changes by a mean of 0.0047 and at most 0.0288, 4.4 percent of that column's standard deviation; the largest relative move in any column MEAN across 60,000 rows is 0.34 percent. Seven figures drawn from that table are redrawn. |
| **Fix** | **Text.** State that the comparison between UQ methods is paired, cite the three papers, and report that pairing changes the measured differences by about 1 percent while removing a 3.7 percent floor under any statement about which material ranks first. `TABLE_CRNComparison.csv`. |
| **Status** | Open, text only. Decision 105. |

## 97. "ECI Rank #1 Frequency" is reported without an interval, and every pLCA output now has one

| | |
|---|---|
| **What the manuscript does** | Reports NRMSE between the six UQ methods for each pLCA output, and the headline rank-1 frequencies, with no uncertainty attached to any of them. |
| **What is now available** | A cluster bootstrap over pLCA GROUPS on every one, because the four materials of a pLCA share its total and its variates and a row bootstrap comes back more than twice too narrow. **`eci_rank_1` has an NRMSE of 1.042 [1.033, 1.051]**; the uncertainty index, which the study computes and reports nowhere, is the lowest of the main outputs at 0.503 [0.491, 0.515]. |
| **Why the level matters and not only the interval** | An NRMSE above 1 means the root mean squared difference between two UQ methods exceeds the standard deviation of that output across every material and method. For the paper's headline output the choice of method moves the answer by more than the spread it is trying to describe. |
| **Fix** | **Text.** Put an interval on every reported NRMSE and percentage, and say what an NRMSE above one means. `TABLE_PLCANRMSE.csv`. |
| **Status** | Open, text only. Decision 110. |

## 98. The number of materials per pLCA: the effect does not dilute, and which output is named decides the sentence

| | |
|---|---|
| **What the manuscript assumes** | Four materials per probabilistic LCA throughout, with no statement about what the results would be in a building with more. |
| **What was measured** | The sweep over 2, 3, 4, 6, 8 and 12 materials, 400 resampled groupings each, at equal intensities. **A material's own estimated contribution does not dilute**: the change caused by switching UQ method goes 0.069, 0.076, 0.081, 0.089, 0.089, 0.095 from two materials to twelve. **Its share of the building total does**, 0.0199 to 0.0077. **And the probability that two methods name a different largest contributor RISES**, 0.448 to 0.656, because more materials means more chances of a near-tie at the top. |
| **Why it matters for the text** | The natural sentence -- "with more materials each one matters less, so the choice of method matters less" -- is true only of the share-of-total outputs. Written without naming the output it is wrong in the direction that flatters the study. |
| **Fix** | **Text**, one paragraph with the three directions and the group sizes they were measured over. `TABLE_PLCAGroupSize.csv`. |
| **Status** | Open, text only. Decision 106. |

## 99. Every material carries a use intensity of 1.0, and the ranking results are an upper bound because of it

| | |
|---|---|
| **What the manuscript does** | Sets the material use intensity of all four materials to 1.0 and normalizes every dataset to a mean of 1.0, so the four contributions are exchangeable and their ranking is as fragile as it can be made. The assumption is not stated as a limitation. |
| **What was measured** | Intensities drawn on the simplex from a symmetric Dirichlet, concentration 200 down to 0.15, plus 1:1, 2:1, 10:1 and 100:1, crossed with the group-size sweep. Reported against the ratio of the largest mean contribution to the second largest, which a practitioner computes from a quantity take-off in one line. **The probability that the choice of UQ method changes the leading material crosses 1 percent at a ratio of 2.13 [2.09, 2.17], 5 percent at 1.64 and 10 percent at 1.46**, with an isotonic fit giving 2.22, 1.61 and 1.35 and the 1 percent crossing moving only from 1.90 at two materials to 2.34 at twelve. |
| **The other half, which is the finding** | **A dominant material does nothing for the numbers.** At four materials the median change in a material's estimated contribution is 0.188 at equal intensities, 0.186 at 2:1, 0.175 at 10:1 and 0.166 at 100:1 -- an 11 percent decline while the flip probability falls from 0.546 to zero. **State the group size with it.** The change in the most-affected material is a maximum over the group and so grows with the group whatever the intensities, 0.109 at two materials and 0.346 at twelve; within a group size dominance moves it little and not always downward. What falls cleanly everywhere is the change in the AVERAGE material, 0.081 to 0.045 at four materials. A figure pooling the group sizes shows a rise that is a group-size effect wearing a dominance label. |
| **The anchor** | Marsh, Lewis, Hattam and Allen (in press) state that for their Concrete-Precast staircase under the ICE recommended factors the top two products are steel bar at **42 percent** and precast concrete at **41 percent**: a top-two ratio of **1.02**. A real building element can sit within a percentage point of a tie, which is exactly where the choice of UQ method decides the ranking. Their per-product quantities are in supplementary material this repository does not hold, so no further ratio is computed and no bill of quantities is reconstructed. |
| **The scope limit that belongs in the same paragraph** | Material use intensity is deterministic within a run. Real quantity take-offs carry their own uncertainty, which in practice can exceed the coefficient uncertainty this paper is about. |
| **Fix** | **Text and a new figure.** State the exchangeable-intensity assumption as a limitation, give the crossing with its interval, give the anchor, and say that the continuous outputs do not depend on the assumption the way the ranking does. `TABLE_PLCARatioCrossings.csv`, `TABLE_PLCARatioCurve.csv`, `TABLE_PLCARatioAnchor.csv`, `CompareUQMethods_FIG_MaterialDominance.png`. |
| **Status** | Open, text and figure. Decision 107. |

## 100. THE ANSWER AGAINST THE TRUTH: the KDE and the lognormal are indistinguishable at the decision level and the normal is 40 percent worse

| | |
|---|---|
| **What the manuscript compares** | How far each fitted CDF sits from a target, and how the six methods differ from each other downstream. Neither says how far the ANSWER is from the right one. |
| **What was measured** | Every pLCA group run twice on the same uniform variates, once with the fitted models and once with the datasets' TRUE parents, which this project can reconstruct by replaying the generator. The difference is the error the fitted model causes, with no Monte Carlo noise in it. 2,500 groups, 10,000 draws, cluster-bootstrap intervals. |
| **The result** | Error in a material's rank-1 frequency: `Lognormal, Uniform` **0.0799** [0.0781, 0.0817], `KDE, Uniform` **0.0815**, `KDE, Variable` **0.0825**, `Lognormal, Variable` **0.0853**, `Normal, Uniform` **0.1152**, `Normal, Variable` **0.1193**. **The four non-normal methods span 6 percent of each other and the normal is 40 percent worse than any of them.** |
| **What no method does** | Recover the answer. The best names the material the truth says is the largest contributor **53 percent** of the time against 25 percent for a coin toss among four, and its error in a material's estimated contribution is **0.12** where every material contributes 1.00. |
| **Why this is the cleanest thing the paper can say** | It converts a goodness-of-fit ranking into a decision-level statement: at the level of the answer, choosing between a kernel estimate and a three-parameter lognormal does not matter, and choosing a normal does. That is a recommendation a practitioner can act on without running the analysis. |
| **The definitional part, kept separate** | Against the SAMPLING parent -- the population a uniform-weighted method is actually estimating -- `KDE, Uniform` scores 0.0609 rather than 0.0815 and the variable-weighted methods get worse. That gap is definitional, not an error of estimation, and both are reported. |
| **The caveat** | Every material carries an intensity of 1.0, so the rank-based figures are an upper bound on how often a method gets the ranking wrong. The contribution error does not have that dependence. |
| **Fix** | **Text and a new figure**, and it should lead the results rather than follow them. `TABLE_PLCATruth.csv.gz`, `TABLE_PLCATruthSummary.csv`, `TABLE_PLCATruthWinShare.csv`, `CompareUQMethods_FIG_PLCATruth.png`. |
| **Status** | Open, text and figure. Decision 109. |

## 101. The flip thresholds are conditional on four materials, and the conditionality is now measured

| | |
|---|---|
| **What the manuscript would say** | That the probability of a changed top contributor crosses 1, 5 and 10 percent at relative Wasserstein distances of 0.0018, 0.011 and 0.025, which were calibrated on groups of four materials. |
| **What was measured** | The same calibration at 2, 3, 4, 6, 8 and 12 materials. The flip RATE rises with the group size, 0.106 to 0.188. The thresholds do not fall as expected: under the MAXIMUM of the per-material distances, which is what the stored constants use, they roughly double from two materials to twelve; under the MEAN they are flat. **The rise is the summary drifting, not the pLCA changing** -- a maximum over twelve materials is drawn from more chances than a maximum over two. |
| **What follows** | The stored constants stand, notebook 1's weighting-risk probabilities are not recomputed, and the manuscript states the conditionality with the measured dependence beside it rather than as a bare caveat. |
| **Fix** | **Text.** `TABLE_FlipCrossingsByGroupSize.csv`. |
| **Status** | Open, text only. Decision 108. |

## 102. RESOLVES entry 87. A smoke run can no longer reach outputs/

| | |
|---|---|
| **What entry 87 asked for** | That notebook 3 redirect every write -- tables AND figures -- to a scratch directory when `COMPAREUQ_SMOKE_COMBOS` is set, rather than writing to `outputs/` at all, because the weaker version protects only the tables and the figures are the part that gets forgotten. |
| **What was done** | Exactly that. Every path notebook 3 writes goes through `OUT`, which smoke mode points at a fresh temporary directory. Verified with an 8-group run that wrote all nine tables and eight figures into a temporary directory and left `outputs/` clean. Two tests hold it in place, one static and one on the committed table's own metadata. |
| **Fix** | **None. Code, done.** |
| **Status** | RESOLVED in Stage 2e. Decision 111. |

## 103. A figure in the paper could not be regenerated from the notebooks

| | |
|---|---|
| **What was found** | Running notebook 3 end to end under the smoke configuration, which had never been done for the cells Stage 2d added, three cells failed. The flip-calibration figure read `dct_empirical`, a name notebook 3 never defines, and called `fitting.fit_kde` when only the names imported FROM `fitting` were in scope; both work in a kernel that has run notebook 2 first, which is how they were written. Separately `plca` was a loop index in notebook 3 long before `src/plca.py` existed, so importing the module left every later call reading an integer. |
| **Why it matters beyond the fix** | The project's own rule is that everything must be traceable back to the notebooks and that they reproduce the entire analysis. `CompareUQMethods_FIG_FlipCalibration.png` is in the deposit and could not be regenerated from them as committed. |
| **Fix** | **Code, done.** The notebook loads the one empirical dataset that figure illustrates, spawning its stream after the calibration's so no Stage 2d number moves, and a test now refuses any notebook variable that takes the name of a module the notebook imports. |
| **Status** | RESOLVED in Stage 2e. Decision 112. |

## 104. Concentration fixes the ranking and leaves the error in the numbers where it was, by two independent routes

| | |
|---|---|
| **Why a second route matters** | Entry 99 reports that concentrating a design's contributions on one material takes the probability of a changed leader from 55 percent to zero while the change in a material's estimated contribution falls only 12 percent. That compares the UQ methods with EACH OTHER. This compares each of them with the RIGHT ANSWER, which is a different measurement and could have disagreed. |
| **What was measured** | The same probabilistic LCAs run against the true parent distributions at three intensity settings, 600 groups each, with the leading material at 1, 2 and 10 times every other. At **10:1 all six methods name the true largest contributor in every group**, and the error in a material's rank-1 frequency falls from 0.078-0.119 to **0.0075-0.0091**. The error in its estimated contribution goes 0.115-0.161, 0.116-0.163, **0.119-0.173** -- it does not move. |
| **Which denominator, and why it is not a choice** | The intensity vector is normalized to a mean of 1.0 in every cell, so the building's total mean contribution is the same number whatever the concentration, and an absolute error of 0.12 is the same share of the building at 1:1 as at 10:1. Read instead as a fraction of the LEADING material's own contribution the same error does fall, because that material is larger. **The text must say which denominator it is using.** |
| **The sentence the paper can write** | A practitioner whose design has one dominant material can trust the ranking under any of these methods and still cannot trust the magnitude, which is what a carbon budget is written in. |
| **Fix** | **Text.** `TABLE_PLCATruthByIntensity.csv`. |
| **Status** | Open, text only. Decision 113. |

## New, found in the Stage 2e review

## 105. The paper reports ranking metrics and should report five statements, of which ranking is one

| | |
|---|---|
| **The author's objection** | The draft led with the error in a material's chance of being the largest contributor. "If something predicts a different material as being first, but first and second are extremely close, the fact that one is over the other doesn't seem very important." The same objection retired the flip rate in entry 94. |
| **The five statements a probabilistic LCA makes** | **Magnitude**: the building total as a distribution, including the chance of meeting a budget. **Attribution**: each material's contribution and share. **Information**: which material's uncertainty dominates, so it is worth measuring better. **Action**: what an intervention delivers and how likely it is to. **Comparison**: whether one design beats another. |
| **Where the study stood** | It computed attribution in full, computed information and reported it nowhere, kept only the mean and standard deviation of the total and of each intervention so the confidence half was missing, and never made a comparison at all. |
| **Fix** | **Text and analysis, both now done.** Three sections -- what the method does to the numbers, to where the uncertainty sits, and to the decision -- covering all five. |
| **Status** | Analysis done in Stage 2e. Text open. Decision 114. |

## 106. The ECC cap was taken from each method's own draws, which forced the strategy's signal to zero

| | |
|---|---|
| **What the code did** | `upper_limit = np.quantile(col, capecc)` on each method's own 10,000 draws. So the six methods were asked about six different interventions -- and because each was capped at its own 75th percentile, **exactly 25 percent of iterations were capped under every method by construction**. |
| **Why that is worse than a comparability problem** | A method that understates the upper tail *should* conclude that capping buys less. Under that form it could not: the quantity the strategy exists to measure was fixed by the construction. |
| **What it is now** | One absolute cap per material, the 75th percentile of the values a specifier holds, applied to every method and to the true parent. It is what a practitioner can compute, and it exists on the empirical arm where no parent does. **The share of iterations capped now runs from 0.277 under `Lognormal, Uniform` to 0.366 under `Normal, Uniform`.** |
| **What moved** | Every `capecc_*` column of the results table. `capecc_red_mean` -0.8623 to -0.8349, `capecc_perc_mean` -0.1712 to -0.1698; the rank-frequency columns move further and for a second reason, entry 107. |
| **Fix** | **Code, done. Text**: the manuscript describes the cap and must describe this one. |
| **Status** | Open, text. Decision 115. |

## 107. RESOLVES the `(1-capecc)` divisor, which the cap fix made wrong

| | |
|---|---|
| **What it was** | A count over all iterations scaled by 1 / 0.25. That was exact only because the old cap bound in exactly 25 percent of iterations for every material. |
| **Why it had to go** | Under an absolute cap the bound share is a property of the material and the method, so the divisor scaled by a number that is no longer the right one and a pLCA's four columns summed to 1.25. |
| **What it is now** | A plain count over the Monte Carlo draws: the share of iterations in which capping this material both bound and gave the largest reduction of the four. Across the four ranks of one material the columns sum to the share of iterations its own cap bound; across the four materials the rank-1 column sums to the share in which any cap bound. |
| **Fix** | **Code, done.** The roadmap gives this metric to Stage 2g, which should still revisit it; what forced the change was that the old divisor was correct only under a cap that no longer exists. |
| **Status** | Partly resolved in Stage 2e. Decision 116. |

## 108. THE DECISION A DESIGNER ACTUALLY MAKES IS NOT SENSITIVE TO THE CHOICE OF UQ METHOD

| | |
|---|---|
| **What the manuscript compares** | One building, four materials, no alternatives. A designer chooses between designs, which is what Heijungs (2021), Prado-Lopez et al. (2014) and Marsh et al. (in press) all measure and what all three are cited for. |
| **What was measured** | 800 pairs of options sharing three materials and **the same random draws for them**, differing in the fourth, with the replacement's use intensity carrying a controlled expected saving. Scored against the same comparison run with the true parents. |
| **The result** | P(option B beats option A), against a claimed saving of 0, 1, 2, 5, 10 and 20 percent of the building: the truth gives **0.505, 0.529, 0.554, 0.629, 0.754, 0.953**, and the spread across the six UQ methods is **0.006, 0.006, 0.007, 0.012, 0.020, 0.012**. **The choice of method changes the stated probability that a substitution is an improvement by at most two percentage points, and every method is within two and a half points of the truth.** |
| **Why it should lead** | Every other comparison in this study is between a method and another method, or between a method and a target it was fitted to. This is the decision, scored against the right answer, and the answer is that the choice is safe. |
| **Fix** | **Text, and it is a new section.** `TABLE_PLCADesignSwap.csv.gz`, `TABLE_PLCADesignSwapSummary.csv`. |
| **Status** | Open, text. Decision 118. |

## 109. Where the choice of method does matter: the upper tail, the budget, and the value of specifying

| | |
|---|---|
| **The building's upper tail** | **Every method understates the building's 90th percentile**, by 0.09 to 0.36 on a four-material building whose total averages 4.0. |
| **The budget statement** | At a budget the truth meets 90.0 percent of the time, `Lognormal, Variable` reports **91.3** percent and `Normal, Uniform` reports **86.8**. A practitioner is told they are safer or less safe than they are, in both directions depending on the method. |
| **The specification policy** | Against a true mean saving of **5.39 percent** of the building, the normal reports 6.19 and the lognormal 4.88. Asked for the chance of achieving at least a 5 percent saving, the truth is **23.2 percent** and the normal says **30.5**, an overstatement of 7.4 points, while the KDE is within 1.3 and the lognormal within 0.3. |
| **The contrast that explains it** | **The quantity strategy is method-independent to four decimal places.** Using 25 percent less of a material is a deterministic fraction of its own contribution, so no distributional assumption enters; specifying a cap acts entirely through the upper tail, which is exactly what the methods disagree about. |
| **Fix** | **Text.** `TABLE_PLCABuildingSummary.csv`, `TABLE_PLCAInterventionSummary.csv`. |
| **Status** | Open, text. Decision 119. |

## 110. The methods fail on the same materials, and the families fail in opposite directions

| | |
|---|---|
| **What was asked** | Whether the different UQ methods fail in the same direction or not. |
| **Direction** | On a material's estimated contribution the normal is biased **high** (+0.044 uniform, +0.050 variable), the lognormal **low** (-0.038, -0.027), and the kernel estimate is nearly unbiased (-0.014, +0.003). On a material's 95th percentile they all fail the same way: every one understates it, the KDE by 0.085 and the normal by 0.193. |
| **Materials** | Per-material errors correlate **0.892 to 0.970 between methods that share a weighting scheme** and only **0.581 to 0.714 across weighting schemes**; all six err in the same direction on **51.2 percent** of materials against about 3 percent if they were independent. |
| **What follows** | The dominant axis of disagreement is the WEIGHTS and not the family, the three families make nearly the same error on the same material, and **choosing a different family does not hedge the risk**. |
| **Fix** | **Text, and it belongs with the figure**, which now shows the six signed-error distributions so bias reads as a shift and imprecision as a width. `CompareUQMethods_FIG_PLCATruth.png`. |
| **Status** | Open, text. Decision 120. |

## 111. Knowing market shares buys 17 percent; guessing them with a flat Dirichlet captures a third of it

| | |
|---|---|
| **The framing constraint, which the author set** | The contrast is between **knowing** market shares and **guessing** them, not between two weighting schemes, and nothing here says uniform weighting is better. |
| **What was measured** | 1,200 pLCA groups against the market-weighted parent, with a third weighting added: the same mode-level market share split EQUALLY inside each mode, which removes the flat-Dirichlet noise the generator introduces and the real world does not have. |
| **The result** | Mean absolute error in a material's estimated contribution: KDE **0.1285** uniform, **0.1215** variable, **0.1064** oracle; lognormal 0.1243, 0.1168, 0.1007; normal 0.1620, 0.1629, 0.1550. On the rank-1 frequency the oracle gains as much again -- KDE 0.0815, 0.0818, 0.0717 -- **and the stand-in gains nothing at all**. |
| **How to say it** | Variable weighting is better than uniform on the magnitude, a wash on the ranking, and would be better than both if the shares were known. What separates the oracle from the realized weights is noise this generator introduces by construction, which is entry 89's finding reaching the decision level. |
| **Fix** | **Text**, and it strengthens the call for real production volumes. `TABLE_PLCAOracleSummary.csv`. |
| **Status** | Open, text. Decision 121. |

## 112. Dispersion enters the safe-lead rule, and the natural way to write it down saturates

| | |
|---|---|
| **The question, asked twice** | Whether the lead a material needs is just the ratio of the means, when the spread should surely matter -- and whether it would be better expressed as how many standard deviations apart the two materials are. |
| **What was wrong the first time** | The flip was regressed on log(ratio) and log(CV) as separate terms. The right quantity is the standardized separation, `(r - 1) / sqrt((r x CV_lead)^2 + CV_second^2)`, and `log(r)` is the wrong numerator near r = 1 where every flip happens, so that test understated dispersion by construction. |
| **What the proper test says** | Dispersion moves the risk at a fixed lead by a factor of two: at a lead of 1.6 to 2.2 the flip rate runs **4.2 percent** for a pair worth 0.3 to 0.6 standard deviations and **2.0 percent** for one worth more than 1.6. At a lead of 2.2 to 3.5 it runs 2.6 percent to 0.1. |
| **And the rule barely moves** | The lead needed for a 1 percent risk is **2.24** at a coefficient of variation of 0.25 and **2.36** at 1.5 -- a six-fold range of dispersion moves it by 5 percent. |
| **Why it cannot be written in standard deviations, which is the part to print** | The measure **saturates**: as the lead grows it tends to `1 / CV_lead`, because the leading material's own spread grows with its size. The median material here has a CV of 0.55, so it can never be more than about **1.8 standard deviations** clear of a smaller one however large its lead; the observed median runs 0.09, 0.76, 1.19, 1.60, 1.84 as the lead goes from 1.2x to over 20x, pinned against its ceiling. A rule in standard deviations could not tell a 10x lead from a 100x one. |
| **Fix** | **Text**, and print the two-way table rather than a coefficient: the risk at a given lead, split by how many standard deviations that lead is worth, with counts. `TABLE_PLCAFlipByLeadAndSpread.csv`, `TABLE_PLCASafeLead.csv`, `TABLE_PLCASeparationCeiling.csv`. |
| **Status** | Open, text. Decision 122. |

## 112b. A small bias per material is a large error for a building, because bias adds and noise does not

| | |
|---|---|
| **The question** | Whether the bias directions of entry 110 are significant or minimal. |
| **The answer** | Minimal per material and decisive per building. On one material a method's bias is about a sixth of its noise -- 0.044 against 0.269 for `Normal, Uniform` -- but summing four materials multiplies the bias by four and the noise by two. |
| **Measured** | Building-level systematic error: `Normal, Variable` **+4.98 percent**, `Normal, Uniform` +4.41, `KDE, Variable` +0.35, `KDE, Uniform` -1.41, `Lognormal, Variable` -2.73, `Lognormal, Uniform` **-3.81**. The implied and observed columns agree to four decimal places because the bias is exactly additive. |
| **What the paper should say** | **The choice of method shifts a whole building's estimate by up to nine percentage points from end to end, systematically, and using more materials will not average it away** -- the same percentages hold for a twenty-material building while the random part falls as one over the square root of the count. |
| **Fix** | **Text.** `TABLE_PLCABias.csv`. |
| **Status** | Open, text. Decision 122b. |

## 113. Every negative tick label this project has drawn was a Unicode minus

| | |
|---|---|
| **What was found** | `FIGURE_STYLE.md` requires plain ASCII and names the Unicode minus explicitly. Nothing had ever set matplotlib's `axes.unicode_minus`, so its default U+2212 went into every figure with a negative axis value. |
| **Fix** | **Code, done**: one line in the style module, and a test that draws a figure and asserts its tick labels are ASCII. It reaches the figures built since the style guide existed; the older figures do not call the style module and are Stage 3's, which owns bringing every figure to the guide. |
| **Status** | RESOLVED in Stage 2e for the figures that use the style module. Decision 123. |

---

## New, found in Stage 2f

## 114. RESOLVES entry 11. The two normality columns were two different statistics, and both are now Shapiro-Francia

| | |
|---|---|
| **What entry 11 recorded** | `customstats.shapiro_wilk_weighted` returned the true Shapiro-Wilk W from scipy when the weights were uniform and a Shapiro-Francia W' when they were not. So `fit_norm_SW` and `fit_norm_SW_uw` came from two different estimators, and the same for the lognormal pair. Figure 4e to 4h present them side by side as one statistic computed two ways. |
| **The decision** | **Shapiro-Francia for both.** It is the only one of the two with a weighted form, so it is the only choice under which the uniform-versus-variable comparison is a comparison of one statistic under two weightings, which is what those four panels claim to show. Author decision. |
| **What the columns are called now** | `fit_norm_SF` and `fit_lognorm_SF`, renamed so the column name says which statistic it holds. The display labels read "Shapiro-Francia". |
| **How much it moved** | Only the two UNIFORM columns can move; the variable-weighted ones were already Shapiro-Francia and are bit-identical. Median absolute change in `fit_norm_SF_uw` on the empirical arm: **0.0103** at n = 3-9 (20 datasets), **0.0062** at n = 10-99, **0.0029** at n = 100-999, **0.00005** above n = 1,000. Arm mean 0.7862 to 0.7834. Largest single change on either arm 0.0358. |
| **The equivalence claim in the old docstring fails where it matters** | It said the two are indistinguishable for n >= 20. At n = 20 the median absolute difference is 0.006 and at n = 2,000 it is 0.0004, so that is about right there -- and the smallest size stratum in this study is **n = 3 to 9**, where it is 0.010 with a maximum of 0.029. The equivalence argument does not hold in the regime the study most depends on. |
| **What did NOT move** | The generator calibration, exactly. The tuning objective reads only variable-weighted columns, so every one of the ten standardized distances is unchanged to the last digit and no generation decision is reopened. No fit, no W1 score and no pLCA number moves either: the Shapiro statistic is a reported characteristic and enters nothing. |
| **Fix** | **Text.** Say Shapiro-Francia, say it is one statistic under two weightings, and give the reason: it is the only one of the two that admits sample weights. `audits/shapiro_estimator.py`, `outputs/tables/audits/AUDIT_ShapiroEstimatorByStratum.csv`. |
| **Status** | RESOLVED in Stage 2f. Decision 125. |

## 115. `_royston_pvalue` was wrong in two ways, not the one that was known

| | |
|---|---|
| **What was known** | The `4 <= n <= 11` branch applied the `n >= 12` polynomials. |
| **What was actually wrong** | **Two separate defects.** (1) That branch applied polynomials in log(n) to a range whose Royston coefficients are polynomials in **n itself**, and subtracted the gamma shift from the transformed variable instead of applying Royston's `-log(gamma - log(1 - W))` re-expression. At n = 10 and 11 it returned **1.0000** where the correct value is about 0.50; the largest observed error was **0.99**. (2) The `n >= 12` branch evaluated its sigma polynomial at **log(log(n))** where Royston evaluates it at **log(n)**, so the p-value was wrong at EVERY sample size, by up to **0.077** at n = 5,000. |
| **What depended on it** | Nothing. Only the statistic is kept, by decision, because a p-value at n = 77,548 measures the sample size rather than the departure from normality. |
| **Fix** | **Code, done.** Both defects corrected; the function now reproduces `scipy.stats.shapiro`'s own p-value to **4e-12** for 4 <= n <= 5000, and a test pins that. Since the statistic the module returns is now Shapiro-Francia, it gets Royston's (1993) W' transform instead, which returns NaN outside 5 <= n_eff <= 5000 rather than extrapolating a fit past the range it was made on. |
| **Why it is in this log at all** | It reaches no number in the paper, but the repository is a public Zenodo deposit and a known-wrong statistical routine in it is a defect a reader can find. |
| **Status** | RESOLVED in Stage 2f. Decision 126. |

## 116. RESOLVES entry 9. The panel count: 18 was wrong, 19 was right then, and it is 21 now

| | |
|---|---|
| **Manuscript** | "the 18 statistical metrics calculated for each synthetic ECC dataset", with panels referenced 4a through 4r. |
| **Confirmed composition** | The figure draws every characteristic both arms carry, except `mean_uw`, which is identically 1.0 by construction because every dataset is divided by its own unweighted mean. Before Stage 2a that was **19**: eight characteristics with a uniform and a variable version (coefficient of variation, entropy, the two Shapiro fits, kurtosis, the modality index, skewness, weight of outliers) = 16 panels, plus **three single panels** -- dataset size, the uniform-to-variable Wasserstein distance, and **the variable-weighted MEAN**, which is the one the earlier count could not name. |
| **So** | **The manuscript's 18 is wrong and 19 was correct for the figure as the manuscript describes it.** |
| **And 19 is no longer correct either** | Stage 2a added Silverman's critical bandwidth under both weightings, so the figure as the code now draws it has **21** panels. |
| **Fix** | **Text, but wait for Stage 2f's reduced figure.** The whole point of the reduction is that 21 marginal panels represent about four to five independent quantities, so the number in the manuscript should be the reduced figure's, not 21. State the full candidate count in the supplement and the survivor count in the main text. |
| **Status** | RESOLVED as a count; the sentence depends on the reduced figure. Decision 127. |

## 117. The visible-mode counts were a 500-dataset sample, which is now the whole corpus

| | |
|---|---|
| **What was found** | Notebook 1 computed `TABLE_VisibleModes.csv` on 500 of 10,000 synthetic datasets, on the assumption that counting modes is expensive. Timed in Stage 2f: all 10,000 take **under a minute**. |
| **Why it mattered** | It kept the two visible-mode counts out of the Stage 2f reduction as first-class predictors: a complete-case model over 402 usable rows of 10,000 drops the corpus, and the complete-case cost the stage was asked to report would have been dominated by that artifact rather than by the undefined kurtosis it is about. |
| **What moves** | The synthetic share with one, two and three or more visible modes, by sampling error only, since the 500 were a random draw. The empirical arm is unchanged in method and now covers every category with n >= 8. Any figure quoting a synthetic mode share must be taken from the rebuilt table. |
| **Fix** | **Code, done.** Notebook 1 computes both counts for every dataset. **Text**: quote the full-corpus numbers. |
| **Status** | RESOLVED in Stage 2f. Decision 128. |

## 118. A characteristic in the metric set IS part of the score it was being compared against

| | |
|---|---|
| **What was found** | Every model in this study is scored against the VARIABLE-weighted empirical CDF, including the three uniform-weighted fits, so a uniform-weighted model carries a distance no estimator can remove. That distance is exactly `w_v_uw_wasserstein`, the Wasserstein distance between the uniform-weighted and variable-weighted versions of the dataset, which the study reports as one of its statistical characteristics. |
| **Measured** | Its Spearman correlation with the definitional term of the score is **1.000000** for all three uniform-weighted methods on both arms. On that identity alone it reaches a Spearman of **0.966** with the in-sample W1 of `KDE, Uniform`, 0.943 for `Lognormal, Uniform` and 0.767 for `Normal, Uniform`. |
| **Why it matters to the text** | Any sentence of the form "W1 rises with the uniform-to-variable distance" is, for a uniform-weighted method, a restatement of the definition of the score rather than a finding about ECC data. The manuscript must not present that panel as a relationship. |
| **What IS a real result** | On the DOWNSTREAM error there is no identity: the variable-weighted methods have a definitional term of exactly zero, and the characteristic still correlates **0.72, 0.72 and 0.57** with the error in a material's estimated contribution under the three uniform methods and 0.56 to 0.59 under the variable ones. So it genuinely predicts how wrong the answer is. |
| **Fix** | **Text**, and **code, done**: the survivor ranking is reported with and without it, and `reduction.definitional_check` flags an exact identity so a later candidate derived from the scoring target cannot slip in unnoticed. |
| **Status** | RESOLVED in Stage 2f. Decision 130. |

## 119. The three modality measures disagree, and the one that predicts predicts nothing of its own

| | |
|---|---|
| **Manuscript** | Reports a modality index among the statistical characteristics, and quotes a share of ECC datasets as unimodal. |
| **They are three different measures** | Spearman between the continuous index and the fitted-bandwidth mode count is **+0.168 on the real arm and +0.018 on the corpus**; between Silverman's critical bandwidth and that count, **+0.283 and -0.115**. A negative correlation settles that they are not measuring one property. |
| **Which one carries signal** | Offered alone over a spline in log(n), Silverman's critical bandwidth adds a mean incremental R2 of **0.210** on the real arm and **0.175** on the corpus, significant on every model. The continuous index adds 0.128 and 0.042. The mode COUNTS add **0.012**, and on the real arm the fitted-bandwidth count is significant on **none** of the twelve models, median p = 0.27. |
| **And it is not its own** | In the full multivariate model the critical bandwidth ranks **14th of 22** and is in the top five of **zero of 96** models, because it correlates **+0.54 and +0.60 with the coefficient of variation**. Everything it appeared to carry over size alone is dispersion it travels with. |
| **How this relates to entry 79** | It CONFIRMS it and explains it. That entry found multimodality last of eleven characteristics; it tested a mode COUNT, which is indeed worthless. The critical bandwidth is not worthless, it is dispersion under another name. |
| **Fix** | **Text.** State that the modality measures disagree, give the correlation, and say that the only one with predictive content is carrying dispersion. Do not present modality as an independent driver. |
| **Status** | Open, text. Decision 132. |

## 120. The corpus has margin where it does not matter and none where it does

| | |
|---|---|
| **Manuscript** | Rests a generalizability claim on the coverage figure: the synthetic datasets span the range of statistical characteristics the empirical datasets occupy. |
| **That claim is true and is not the one that matters** | 98 to 100 percent of real datasets sit inside the synthetic range on every metric, so INTERPOLATION is supported. What supports generalizing past the categories EC3 happens to hold is the MARGIN beyond the empirical range, and that is where the two goals pull apart. |
| **Measured** | `margin_above`, in units of the empirical range, for the five characteristics that carry the signal: coefficient of variation **-0.629**, its uniform-weighted twin **-0.656**, dataset size **-0.678**, entropy +0.126 and +0.132. For three that carry none: skewness **+7.186**, modality index **+6.864**, kurtosis **+5.249**. Median margin **-0.629 for the survivors and +0.509 for the other seventeen**; Spearman between importance rank and margin **+0.484**. |
| **How this relates to entry 34 and the coverage decision** | It sharpens rather than reverses it. That decision accepted the dispersion shortfall on the grounds that what the corpus cannot reach is the shape of a contaminated EC3 category rather than of a material, which is an argument about WHICH datasets are uncovered. This adds that the shortfall sits on the single most predictive characteristic in the study, which bounds how far the conclusions carry regardless of which categories are uncovered. |
| **Fix** | **Text.** State the limitation in these terms: the study's conclusions are supported across the range of dispersion and dataset size that real EC3 categories occupy, and are not supported beyond it. Generation stays closed. |
| **Status** | Open, text. Decision 133. |

## 121. The characteristic figure is 21 marginal panels of about four independent quantities

| | |
|---|---|
| **Manuscript** | Presents goodness-of-fit against each statistical metric as a rolling average, one panel per metric, and discusses the panels individually. |
| **Three defects in that presentation** | It carries no uncertainty band, so a wiggle and a result look the same; it shows no data density, so a curve through four datasets in a sparse tail looks like one through four hundred; and the metrics are correlated, so the marginal panels overstate how many independent effects exist. |
| **Measured** | The effective dimension of the 23 candidates -- the participation ratio of the correlation eigenvalues, which would be 23 if they were independent -- is **4.33 on the empirical arm and 5.43 on the synthetic**. Seven empirical pairs correlate above 0.9, the worst being entropy against its uniform-weighted twin at 0.991 and the coefficient of variation against its twin at 0.979. |
| **And the marginal view is misleading in a specific direction** | Holding the other characteristics fixed, dataset size keeps **0.73 to 0.82** of its marginal slope and the coefficient of variation **0.43 to 0.66**, while **entropy -- which has the STEEPEST marginal curve of the five survivors, 4.05 log units -- keeps 0.15**. The characteristic whose panel looks most impressive is the one that is almost entirely borrowed. |
| **Fix** | **Text and figure.** Report the survivors with bootstrap bands and a density rug; put the full candidate set in the supplement; and give the effective dimension, which is the number that justifies the cut. |
| **Status** | Open, text and figure. Decisions 129 and 131. |

## 122. Two exclusions the models had to be told about, and the smallest datasets are both

| | |
|---|---|
| **What was found** | Two different things remove the smallest size band from an analysis, and neither announces itself. |
| **Undefined characteristics** | Unbiased excess kurtosis divides by (n-1)(n-2)(n-3), so it is undefined below n = 4: **6 of the 20** real categories with 3 to 9 EPDs and **612 of 2,500** synthetic ones. A visible-mode count needs at least 8 values: **17 of 20** and **2,026 of 2,500**. Every other band is complete on every characteristic. A model that dropped incomplete rows would therefore discard **85 percent of the real n = 3-9 band and 81 percent of the synthetic one** while leaving the rest untouched -- which is precisely the regime where a parametric family is expected to beat a kernel estimate. |
| **An undefined TARGET, which the first check cannot see** | The cross-validated empirical score is undefined below n = 10, because half of a nine-value dataset is four values. So the out-of-sample reduction on the real arm uses **127 of 147** categories and **0 of the 20** in the smallest band. The in-sample target covers all 147 and the two have to be read together. |
| **Fix** | **Code, done**: the additive model imputes with a missingness indicator and the boosted model splits on missingness natively, so both keep every row, and two tables report what would otherwise have been lost, per arm and per band. **Text**: any size-banded claim on the real arm below n = 10 comes from the in-sample target only, and must say so. |
| **Status** | RESOLVED in code; a text note is owed. Decision 129. |

## New, found in the Stage 2f review

## 123. The modality measure the paper should report is the author's own, at the bandwidth the study fits

| | |
|---|---|
| **Manuscript** | Reports a modality index among the statistical characteristics, computed as the summed heights of a kernel density's local maxima less the summed heights of its local minima, over the tallest peak. |
| **What was wrong** | Not the measure. The code computes it at **Scott's rule**, which is what the study used when the function was written; the study moved to a guarded Silverman bandwidth in Stage 2b and nothing brought this measure with it. Four stages then measured the idea at a bandwidth the study had abandoned. |
| **What the bandwidth was worth** | Predicting which of the kernel estimate and the three-parameter lognormal fits better on the 127 real categories, with dataset size AND dispersion already in the model: the index at the **fitted** bandwidth adds an incremental R2 of **0.1114 at p = 0.0002**, which is **2nd of the 23 characteristics tested**; the same index at **Scott's** bandwidth adds **0.0179 at p = 0.69**, which is **21st of 23**. Nothing about the measure changed but the smoothing. |
| **And the replacement was worse than what it replaced** | The visible mode COUNT adopted in its place is the **worst of all 23** candidates at either bandwidth -- 0.0100 at p = 0.33. Counting modes discards exactly the information that subtracting the minima preserves. |
| **A reversal the text must carry** | The reason given for setting this metric aside was that it "spans only 1.000 to 1.159, so read as a count it is constant at 1". That range is Scott's oversmoothing: at the bandwidth the study fits the same index spans **1.000 to 1.249** on the real arm. The readout was never the defect. |
| **Fix** | **Code, done**: the bandwidth is an argument and `modality_index_fitted` is computed beside the untouched original. **Text**: report the index at the fitted bandwidth, say which bandwidth, and do not describe modality as carrying no information. |
| **Status** | RESOLVED in code; text owed. Decision 134, superseding decision 132 and reversing decision 23. |

## 124. The metric reduction answered the level of the score, not which method to use

| | |
|---|---|
| **What was found** | The reduction ranked characteristics by how well they predict the LEVEL of a single method's goodness-of-fit score. The level is dominated by dispersion and dataset size because **every** method gets worse on spread data and on small samples. The paper asks which method to USE, which is the DIFFERENCE between two of them -- and a difference is about whose shape assumption fits. |
| **What the right target says** | Predicting `log(W1_KDE / W1_lognormal)` within a weighting scheme on the real categories, with size and dispersion already in the model (base R2 0.483 and 0.513, 127 datasets, Bonferroni threshold 0.0022 for 23 tests). Uniform weights: the variable-weighted mean **0.125**, the modality index at the fitted bandwidth **0.111**, Silverman's critical bandwidth **0.107**, the weight of outliers **0.105**, kurtosis **0.103**. Variable weights: the lognormal Shapiro statistic **0.140**, its uniform twin 0.103, the critical bandwidth 0.091, skewness **0.077**. |
| **Skewness specifically** | Real, as a reader would expect of a comparison involving a lognormal, and **not the strongest**: 0.077 at p = 0.0041, just past the corrected threshold. The sign is the expected one -- the more right-skewed the data, the better the lognormal does relative to the kernel estimate, because a lognormal is a right-skewed family. |
| **Left skew** | On the corpus, where 1,674 of 10,000 datasets are left skewed, they are harder for all six methods (mean score 0.24 to 0.29 against 0.14 to 0.23). **The real arm has only 8 left-skewed categories of 147**, so it cannot support a claim. Real ECC data is almost never left skewed, and that asymmetry belongs in the text. |
| **Why the ratio is the right target and not a convenience** | Taken within a weighting scheme it CANCELS the part of the score no estimator can remove, so the uniform-to-variable distance becomes a legitimate predictor where on the level it was an identity (entry 118). And it is scale free, so it does not inherit the level's dependence on dispersion. |
| **Fix** | **Text.** Report the choice target as the reduction's result, keep the level and the downstream error beside it because the contrast between the three is the finding, and state that other characteristics were tested and where they fell. |
| **Status** | Open, text. Decision 135, narrowing decisions 129, 131 and 132. |

## 125. The corpus is the weaker arm for the question the paper asks

| | |
|---|---|
| **What was found** | Every incremental contribution on the choice target is an order of magnitude smaller on the synthetic arm than on the real one: **0.01 to 0.03 against 0.08 to 0.14**. |
| **Why** | The corpus does not span the shape variety the real categories do. In real units its coefficient of variation reaches **2.58** against the real arm's **6.93**, its uniform-weighted twin 2.50 against 7.24, and its dataset size stops at **9,978** against **31,025**. Kurtosis and skewness are short by only 1.08x and 1.25x, which is noise. |
| **What it is and is not** | It is a loss of statistical power, not a bias: the corpus covers 98 to 100 percent of real datasets on every characteristic, so interpolation is supported. It means the corpus cannot confirm an effect that the real data show, which is the opposite of the usual worry about synthetic data flattering a method. |
| **Fix** | **Author decision, not a text edit.** Closing the dispersion gap is a generator redesign -- heavier-tailed parents or a different truncation rule -- because eight candidate parameters were measured and none took the achieved coefficient of variation above 2.15. The size cap is a cost decision: the corpus stops at 9,999 by construction and only the three largest ReadyMix strength classes exceed it. Generation is currently closed. |
| **WITHDRAWN 2026-09-21** | **Both halves of this entry are wrong and it must not be quoted.** The real arm's 0.08 to 0.14 increments were IN-SAMPLE R2 on 127 datasets, where adding a five-knot spline buys about 0.043 under the null; measured out of sample the same arm has a base R2 of **-0.724** under variable weighting and only 9 of 46 gains exceed their own fold spread, so it was never measuring anything. And the dispersion shortfall is **one category** -- `Aggregates`, coefficient of variation 6.929, the contaminated EC3 bin -- so it cannot be what limits the corpus. Excising every real category above the corpus maximum moves the kernel estimate's win share from 40.2 to 40.5 percent and the size crossover from n = 124 to 122. |
| **What is true instead** | The synthetic increments are small because they are **correctly measured**, not because the corpus is short. The corpus detects a large effect when one exists: the uniform-to-variable Wasserstein distance gains **+0.170 (sd 0.027)** out of sample. |
| **Status** | WITHDRAWN. Superseded by decisions 136, 137 and 138 and by entries 126 and 127. |


## 126. Every ranking of characteristics taken from the 127 real categories was in-sample

| | |
|---|---|
| **What was found** | The reduction's rankings, including the modality result of entry 123 and the choice-target result of entry 124, were incremental **in-sample** R2 on 127 datasets. A five-knot spline added to 127 points raises in-sample R2 by about **0.043 under the null**, so reported increments of 0.10 to 0.14 are part signal and part arithmetic, and nothing in the numbers distinguished them. |
| **The same data out of sample** | Five-fold cross-validated: the base model of size and dispersion has an R2 of **-0.724** under variable weighting, meaning it predicts worse than the mean; the median gain is **0.032** against a median fold-to-fold spread of **0.242**; **9 of 46** gains exceed their own fold spread. The arm cannot support this model. |
| **What the 10,000 synthetic datasets say instead** | Out-of-sample gain over size and dispersion, fold spread beside it: the uniform-to-variable Wasserstein distance **+0.170 (0.027)** under uniform weighting; then, under variable weighting, the variable-weighted mean **+0.034 (0.007)**, the weight of outliers **+0.026 (0.009)**, the visible mode COUNT at the fitted bandwidth **+0.019 (0.010)** and kurtosis **+0.018 (0.012)**. The author's modality INDEX gains **+0.0015 and +0.0027**, indistinguishable from zero. |
| **What this does to entry 123** | The bandwidth defect stands and the fix stands: the index was hardcoded to Scott's rule after the study moved to a guarded Silverman. The CLAIM built on it does not. Out of sample the index adds nothing, and the mode COUNT that entry 123 called the worst of 23 is the modality measure that survives selection. |
| **Significance** | No p-value appears in any figure, table or claim. A characteristic is reported with its cross-validated gain and the spread of that gain across folds, so the reader sees the effect and its own noise in the same line. |
| **Redundancy** | Handled by forward selection rather than by pruning correlated columns: only what still helps once everything chosen is in. Uniform weights keep the uniform-to-variable distance, the mode count, dispersion, entropy and outlier weight, R2 0.280 to 0.581; variable weights keep the mean, outlier weight, the mode count, dispersion, the normal fit and skewness, R2 0.524 to 0.634. |
| **Fix** | **Text and figures.** Every claim about which method to use is measured on the 10,000 synthetic datasets. The real arm appears as a consistency check with its interval shown and is stated to be too small to confirm anything. |
| **Status** | Open, text and figures. Decision 136. |

## 127. The dispersion shortfall is one contaminated category and the generator cannot close it cheaply

| | |
|---|---|
| **What was asked** | Widen the synthetic datasets' dispersion, on the grounds that the corpus reaches a coefficient of variation of 2.58 where real categories reach 6.93. |
| **Where the dispersion is actually lost** | Not in the draw and not in the finite sample. The drawn TARGET has a median of 1.26 and a p99 of 13.93; the solved PARENT tops out at **1.31**, because **60 percent of targets come back `clipped_max_cv`**. The binding constraint is the truncation, whose width is capped through `1 + 1/min_q1_over_iqr = 3`. The earlier audit swept the target and the floor and **never swept the truncation multiple**, so "not reachable by any parameter" was measured with the binding parameter held fixed. |
| **It is reachable, and the price** | Eight configurations, 440 datasets each, full calibration objective, against a seed-to-seed standard deviation of 0.0066. The best dispersion match halves the coefficient-of-variation distance (0.380 to 0.188) and reaches a maximum of 3.52 -- while multiplying the **uniform-to-variable Wasserstein distance by 2.4** (0.275 to 0.661), doubling Silverman's critical bandwidth and worsening the overall objective by **6.3 seed standard deviations**. It still puts only 0.2 percent of datasets above a coefficient of variation of 2, against the real arm's 4.1 percent. |
| **The gap, by name** | Exactly **1 of 147** real categories exceeds the corpus maximum of 2.576: **`Aggregates`**, n = 378, coefficient of variation **6.929**, skewness 12.5, kurtosis 163.9. Six categories exceed 2.0. `Aggregates` is the contaminated EC3 bin where a relative outlier filter sets its upper bound at 41,238,610 times the median and trims nothing. |
| **Nothing depends on it** | Removing every category above the corpus maximum: the kernel estimate is closest on 40.2 to **40.5** percent under uniform weighting and 34.6 to **34.9** under variable; the size crossover moves from n = 124.2 to **122.1** and from 204.0 to **196.3**. |
| **Fix** | **Text.** State the limitation as what it is: the corpus spans the dispersion of every real MATERIAL category and does not span one contaminated EC3 bin, which is a statement about EC3's taxonomy rather than about how far the conclusions carry. Do not repeat the 2.58-against-6.93 ratio without naming the single category that sets it. |
| **Status** | Open, text; the widening itself is **declined on the measurement and awaiting the author**, who asked for it. Decision 138, narrowing decision 133. |

## 128. The practitioner rule, and the precision it does and does not have

| | |
|---|---|
| **What the paper should say** | Use a kernel density estimate above **75 to 100** declarations in a category, with market-share weights where they are known, and a three-parameter lognormal below. |
| **The cutoff, measured** | **The best threshold is 81 declarations and every value from 68 to 97 is indistinguishable from it**, by a paired bootstrap over datasets: each threshold's excess cost against whichever threshold won on that same resample. Eight independent streams return all three numbers exactly. The policy costs 38.3 percent more error than the best choice that could be made per dataset, against 53.6 percent for always using a kernel estimate and 178.5 for always using a lognormal. **An earlier version of this entry said 59 to 134 and that was too wide**: it came from a tolerance chosen rather than measured, with no uncertainty in it, and at 138 the penalty is 0.79 points with an interval of 0.16 to 1.58, which excludes zero. |
| **Two ranges, two questions** | The kernel and lognormal FAMILIES change places at **46 to 70** declarations, a property of the win-share curves, and that one drifts a few either way on a reseed. The best place for a RULE is **68 to 97**, above the crossing because the penalty is steeper on the high side. The manuscript must not quote either for the other. |
| **Which method actually wins, by size** | Share of datasets on which each is closest to the truth. n = 3-9: kernel estimate with equal weights leads at 33.8 percent. n = 10-99: lognormal with equal weights leads at 25.2. n = 100-999: kernel estimate with market-share weights leads at 42.8. n = 1000+: the same at **69.6**. There is no method that is best regardless. |
| **The curve is U-shaped and the dip is real** | The kernel estimate is closer on 62.6 percent of datasets at n = 3-9, dips below half between about 10 and 55, then rises to 86.5 percent under equal weights and 96.8 under market-share weights. With three to nine values there is no shape to estimate and both families do equally badly; between ten and fifty the lognormal's shape assumption is worth more than the kernel's flexibility. The figure draws the dip rather than smoothing it. |
| **No other characteristic gives a threshold** | Sweeping every candidate for a crossing: the modality index never crosses under equal weights (60.9 to 55.9 percent as it rises) and crosses **downward** under market-share weights (67.0 to 46.5), so more modality makes the kernel estimate relatively worse. Silverman's critical bandwidth is flat, 77.5 to 76.0. This confirms the earlier "nothing but dataset size belongs in the rule" finding on 10,000 datasets out of sample where it had 127 in sample. |
| **Fix** | **Text and figures.** State the rule, state the basin, and do not offer a second threshold on any other characteristic. |
| **Status** | Open, text. Decision 139. |

## 129. Market-share weighting pays when the shares are concentrated, not when they are even

| | |
|---|---|
| **What was found** | Splitting the paired comparison inside a size band by how concentrated the weight vector is (Kish effective sample size over n), at n = 100-999: the market-share kernel fit beats its uniform twin on **74.4** percent of datasets where the weights are most concentrated and **29.8** percent where they are most even. The lognormal gives 79.4 and 37.9. |
| **Why it matters** | It inverts the usual intuition. Concentration is normally read as a shrunken sample and therefore a cost; that captures only the variance half. A nearly even weight vector carries **no information about the market**, so the market-share fit is the uniform fit plus noise and loses. This is the clean statement of what variable weighting is for. |
| **The small-n exception, and it is a different mechanism** | At n = 3-9 the market-share fit wins only 39.0 percent [37.2, 40.9] and the same concentration split is **flat** (41.4, 36.5, 36.3, 41.9). A flat Dirichlet over three to nine points leaves a median Kish effective sample size of **2.7**, with 93.1 percent below five. Nothing about the weights rescues it because the problem is the point count. |
| **How it relates to the block-structure finding** | The same mechanism from the other side: at matched effective sample size, share concentrated on products with adjacent coefficients moves the answer 1.5 to 3.1 times as much as share concentrated at random. What pays is the weights being informative, not their being many. |
| **The target is doing real work and the paper should show it** | Share of datasets above n = 1,000 where the market-share kernel fit beats its uniform twin: **75.8 percent** against the market-weighted parent, **20.7 percent** against the parent each method separately estimates, **98.2 percent** against the in-sample variable-weighted data. Only the first answers the question; printing all three is the clearest way to show why. |
| **Fix** | **Text.** Replace any bare "variable weighting is better above n = 100" with the conditional version, and state the three-target contrast once. |
| **Status** | Open, text. Decision 140. |

## 130. The two arms draw market-share weights by different rules

| | |
|---|---|
| **What was found** | The empirical arm draws a flat Dirichlet over all points; the synthetic arm gives each point its mode's market share split within the mode. So the weights are correlated with the values on one arm and independent of them on the other, on the dimension the paper is about. |
| **Why it is not a detail** | Weights drawn independently of the values are exchangeable, so the weighted CDF converges to the unweighted one and the measured weighting effect must decay like n to the minus one half. Correlated weights do not decay. Measured decay slope on log(n): **-0.397 empirical against -0.167 synthetic**, and at n >= 1000 the median separation is **0.0049 empirical against 0.0501 synthetic** -- a factor of ten. Reweighting the corpus's OWN values flat gives 0.0116, which proves the gap is the weight rule and not the data. |
| **Why nobody noticed** | The two agree in the aggregate because 78 of 147 real categories are n = 10-99, the one band where the two rules give similar answers (0.109 against 0.125). The disagreement is at the ends. |
| **What a fix looks like** | One rule on both arms. Not a fitted mixture: it cannot be estimated at n = 3-9, mode counts on real data are badly method-dependent, and it would put a researcher degree of freedom inside the central quantity. Contiguous blocks over the sorted values do the same job -- what mode coupling produces is weights correlated with values, and the mode structure is only the device. A coherence parameter from random to contiguous membership turns the untestable assumption into a reported axis. |
| **The size of the choice** | Median separation on the real categories at n >= 1000 runs **0.0075 at zero coherence to 0.1219 at full coherence**, against 0.0049 today. A larger separation is not by itself evidence of a better model: it measures what unknown shares do rather than being a target. |
| **But zero coherence is not the neutral option** | A flat Dirichlet is not the absence of an assumption; it is the claim that market share is uncorrelated with carbon intensity, and the evidence contradicts it -- the 63.75 percent Rest-of-World BOF share sits on the HIGHER-carbon steel route while the lower-carbon EAF route is the small one. The decay is also a property of the model rather than of markets: weights drawn independently of the values must converge to uniform as n grows, because that is what exchangeability means, whereas real market share does not become more even as more manufacturers publish. **So reporting "weighting stops mattering at large n" would be reporting an artifact of the weight model.** |
| **Two confounds to control** | The block model changes concentration as well as coherence, so the number of blocks must be matched or swept alongside, as entry 89 matched the effective sample size. And the empirical arm's decay is partly dispersion: its categories above n = 1,000 are the ReadyMix strength classes at a median coefficient of variation of 0.297 against 0.954 at n = 100-999, and separation scales with that. |
| **Fix** | **Stage 2h**, as its first item. Nothing is changed here, no number moves, and the manuscript owes the limitation stated plainly until then: the two arms' weighting effects are not the same quantity. |
| **Status** | Open, owned by 2h. Decision 141. |

## New, found in Stage 2g

## 131. The headline metric recovers the truth worst of every candidate tested

| | |
|---|---|
| **What the manuscript reports** | "ECI Rank #1 Frequency", the share of Monte Carlo iterations in which a dataset is the largest contributor, as the principal downstream result. |
| **The question that could not be asked before** | Until the probabilistic LCA was run against the datasets' TRUE parent distributions, a metric could only be judged on whether the six methods agreed about it. Now it can be judged on whether a fitted model gets it RIGHT. |
| **The statistic** | Mean absolute error against the true parent, divided by the standard deviation of the TRUE value across every material. Below 1 a method's error is smaller than the between-material differences the metric exists to reveal; at or above 1 the metric cannot distinguish two materials at all. It is the same division as the study's own NRMSE on a different numerator, so the two can be read together. |
| **The result, best method to worst** | spread of a material's contribution **0.42 to 0.50**; the 95th percentile of it 0.45 to 0.51; its estimated contribution 0.51 to 0.71; the uncertainty index 0.51 to 0.53; its share at the BUILDING's 95th percentile 0.61 to 0.69; its mean share of the total 0.66 to 0.88; **its chance of leading 0.72 to 1.07**. The study's headline is last on both ends, and **under a normal fit it exceeds 1.0** (1.037 and 1.074). |
| **And which method looks best depends on the metric** | On a win share against the truth, `KDE, Variable` leads on the chance of leading, the 95th percentile and the spread; `Lognormal, Variable` on the mean contribution, the mean share and the share at the building's 95th; `Lognormal, Uniform` on the uncertainty index. **Three different methods across seven metrics**, and the rank correlation of the six methods' ordering with the rank metric's ordering runs from +1.00 to **-0.54**. |
| **Fix** | **Text and figures.** Demote the rank metric to one statistic among five, reported with its noise floor. Lead the attribution section with a material's estimated contribution and carry its share at the building's 95th percentile beside it. Do not state a "best method" without naming the metric it is best on. |
| **Status** | Open, text. Decisions 143 and 144. |

## 132. The results section has an order, and the decision leads it

| | |
|---|---|
| **The order, measured rather than argued** | **1. The design comparison**, because it is the decision a designer makes and the answer is a null: over 800 option pairs against the truth, the choice of method changes the stated probability that a substitution is an improvement by at most **0.020** and every method lands within **0.026** of the truth; at a claimed 5 percent saving the truth is 0.629 and the six span 0.630 to 0.642. **2. The safe-lead rule**: the chance that the method changes which material leads crosses 1 percent at a top-two contribution ratio of **2.13** [2.09, 2.17], and the one real building element available sits at **1.02**. **3. The building total and the budget**: every method understates the total's 90th percentile, by **0.087 to 0.356** on a building averaging 4.0, and at a budget the truth meets 90.0 percent of the time the six report **86.8 to 91.3**. **4. The specification result**: against a true mean saving of **5.39 percent** the six report 4.88 to 6.19, and asked for the chance of at least 5 percent the truth is **23.2** and the six span **22.9 to 30.5**. **5. Where the uncertainty sits.** |
| **The contrast that explains the stage** | A QUANTITY reduction delivers 0.0625 of the building under every method and under the truth, identical to four decimal places, because it is a deterministic fraction of a material's own contribution; a specification cap acts entirely through the upper tail, which is exactly what the methods disagree about. |
| **Fix** | **Text.** This is the results section's order. The attribution metrics are one of the five, not the frame. |
| **Status** | Open, text. Decision 144, confirming the Stage 2e recommendation by measurement. |

## 133. The uncertainty index is the steadiest output and no method recovers it well

| | |
|---|---|
| **What the manuscript does** | Computes the uncertainty index and reports it in no table, figure or section. |
| **The case for reporting it** | Its NRMSE between the six methods is **0.5035** [0.4918, 0.5158] against **1.042** for a material's chance of leading, the lowest of any main output; and asked which material's uncertainty dominates, the best method names the truth's answer **58.4 percent** of the time against a one-in-four chance level, which is the highest decision agreement of any candidate. It is also the fourth reduction strategy under another name: three strategies reduce the expected impact -- use less, specify better, substitute -- and this one reduces the VARIANCE of the answer, which is what obtaining a supplier-specific declaration buys. |
| **The caveat that must travel with it** | **Every one of the six methods is out by about half the metric's own between-material spread**, 0.508 to 0.531, a span of only 4.4 percent from best to worst. So the choice of method genuinely does not matter for it and no method gets it right. A low NRMSE beside a high recovery error is a metric every method agrees on and every method is wrong about, and reporting only the first half would be the most misleading thing this study could do. |
| **And it is not tail-immune** | It is a variance share, so one enormous material takes all the variance: a thousandth of a model's mass at a thousand times the dataset mean moves it by **145 percent**. |
| **Fix** | **Text.** Report it, with the recovery caveat in the same paragraph as the stability claim. |
| **Status** | Open, text. Decision 146. |

## 134. RESOLVES entry 7. The (1-capecc) divisor, and what the manuscript's definition has to say instead

| | |
|---|---|
| **Manuscript** | "EC Reduction [Cap ECC / Mat Red] Rank #i Frequency ... represents the percentage of iterations in which a given dataset is 1st, 2nd, 3rd, and 4th." A plain percentage of iterations, and the definition is stated as equivalent for the two strategies. |
| **The history** | The divisor was a constant 1 / 0.25, exact only while the specification cap was each METHOD'S OWN 75th percentile and therefore bound in exactly a quarter of iterations for every material by construction. Stage 2e made the cap an absolute value per material and had to drop the divisor with it, leaving a plain count whose four columns summed to between **0.31 and 1.00** rather than to 1. |
| **The correct normalization** | Divide by the iterations in which the strategy APPLIES. In an iteration where no cap binds there is no best material to cap, and counting those iterations against all four makes the column depend on how often the strategy applies rather than on which material is right. `capecc_rank_1` now sums to **exactly 1.0** across the materials of a pLCA, and across the four ranks of one material it sums to the share of applicable iterations in which that material's own cap bound. |
| **The applicability is reported, not divided away** | It is signal. The share of iterations in which any cap binds runs from **0.727 under `Lognormal, Uniform` to 0.838 under `Normal, Uniform`**, with an NRMSE between methods of **1.204**, the highest of any cap column: a method that puts more mass above the cap finds it binding more often, and the old constant forced that to 0.25 for every method. |
| **What moved** | Only the four `capecc_rank_*` columns; 36 of the 40 shared columns of the results table are bit-identical. `capecc_rank_1` mean **0.1929 to 0.2500**, rank 2 0.0922 to 0.1167, rank 3 0.0238 to 0.0294, rank 4 0.00255 to 0.00306. |
| **Fix** | **Text.** The two strategies are NOT normalized the same way and the manuscript says they are. The quantity reduction applies in every iteration and its columns are a percentage of all of them; the cap's are a percentage of the iterations in which capping something helps, and the applicability is a reported quantity beside them. One sentence in the supplementary definitions. |
| **Status** | Entry 7 RESOLVED. Text edit outstanding. Decision 147, completing 116. |

## 135. "The normal is forty percent worse" is a statement about attribution, not about everything

| | |
|---|---|
| **What was claimed** | On the run against the true parents, the four non-normal methods span 7 percent of each other and the normal is 40 percent worse than any of them. |
| **What the companions say** | How much worse the better normal fit is than the best non-normal method, on recovery against the true parent: a material's chance of leading **44.2** percent, its estimated contribution 36.5, its mean share 27.6, the spread of its contribution 17.9 -- and then the 95th percentile of its contribution **4.4**, its share at the building's 95th **2.8**, and the uncertainty index **1.0**. **On the last three the normal is not the worst method at all.** |
| **The ordering is not stable either** | Rank correlation of the six methods' ordering with the ordering the chance of leading gives: +1.00 on the mean share, +0.77 on the spread, +0.49 on the mean contribution, and **-0.26, -0.31 and -0.54** on the 95th percentile, the share at the building's 95th and the uncertainty index. |
| **What DOES survive every metric** | The BIAS. On the three metrics where a signed error is informative -- the mean contribution, the 95th percentile, the spread -- the normal is the most biased of the six on all three, by **+0.21, -0.26 and -0.44** in units of the metric's own spread against the kernel estimate's +0.01, -0.11 and -0.22. Bias adds across the materials of a building while noise cancels. **On a share or a rank frequency the signed error is identically zero by construction**, because the four values sum to one, so that column says nothing about those metrics and must not be read as evidence of unbiasedness. |
| **Fix** | **Text.** "Do not fit a normal distribution" is right for attribution and for anything a building total is summed from. Name the statement the claim applies to, and note that on the tail and information metrics the normal is as accurate as anything and merely more biased. |
| **Status** | Open, text. Decision 148, narrowing 109. |

## 136. The tail a goodness-of-fit score cannot see, and which metrics survive it

| | |
|---|---|
| **The failure mode** | A statistic between CDFs is nearly blind to tail mass and a Monte Carlo is not, because it samples. A model that scores well while carrying a thin enormous tail would dominate any probabilistic LCA it enters. |
| **How blind, measured** | One material of a real pLCA group has a thousandth of its fitted model's mass moved out, with the other three models, the variates and the group held. W1 taken over the scoring grid alone reads **0.262242** at ten times the dataset mean, **0.262571** at a hundred and **0.262571** at a thousand, against an uncontaminated 0.254474. The grid's top is `max(x) + 10 sd`, of order ten times the mean on a normalized dataset, so **beyond it the criterion cannot tell a hundred times the mean from a thousand to six decimal places**. |
| **What closes it** | The tail term added in Stage 2c, which integrates the model's survival function beyond the grid. The same three read **0.262718, 0.353166 and 1.257643**. |
| **Which metrics survive** | Relative change under the same contamination, at ten times the mean and at a thousand: the spread of a material's contribution **0.047 to 34.3**, the uncertainty index 0.051 to 1.45, its estimated contribution 0.006 to 0.66 -- against its share at the building's 95th percentile **0.0082 to 0.0082**, its mean share 0.0017 to 0.0024 and its chance of leading **0.0013 to 0.0013**. A share and a rank frequency saturate; a level has no ceiling, and the spread moves by a factor of 115 in the worst case measured. |
| **The tension this creates, and it belongs in the text** | The metrics that recover the truth BEST are levels, and levels are exactly what a thin far tail wrecks; the metrics that are immune are shares, and they recover worse. **What makes the levels safe to report is that the study's criterion now charges for the thing that wrecks them.** So the tail term is not an accuracy refinement, it is the guard under every level metric the paper reports. |
| **Fix** | **Text.** State the guard where the level metrics are reported. **Stage 2h must keep the tail term in force and report the fitted-model spread ratio at every value of the profile-likelihood guard it sweeps.** |
| **Status** | Open, text, plus a constraint on Stage 2h. Decision 149. |

## 137. Notebook 3 was still explaining pLCA outcomes with the retired in-sample target

| | |
|---|---|
| **What the code did** | One cell scored every fitted model by W1 against the variable-weighted empirical CDF of the values it had been fitted to, and the cell below explained every pLCA outcome against that number. The target is circular twice over: it is the training data, and it is the variable-weighted curve, so a variable-weighted method is scored against itself. |
| **What it does now** | Reads `w1_market`, the score against the market-weighted true parent, from the table notebook 2 writes. That is the population a probabilistic LCA of what gets built is a statement about and the only target under which all six methods estimate the same thing. The retired score is kept beside it, unused, and the cell prints how far the two disagree about which method is closest. |
| **Fix** | **Code, done.** Any figure or sentence drawn from that section must be taken from the rerun. |
| **Status** | RESOLVED in Stage 2g. Decision 150. |

## 138. A break that would have killed any full run of notebook 3

| | |
|---|---|
| **What happened** | One cell iterated the metric LABEL file rather than the frame's own columns. Stage 2f's review added `modality_index_fitted` and its uniform-weighted twin to that file and to the corpus, but not to `corpus.METRIC_COLUMNS`, which is what carries characteristics into the per-dataset dictionary notebook 3 reads. Notebook 3 had not been run since, so the first full run after that stage died there twenty minutes in. Two further labels, `modes_fitted` and `modes_scipy_default`, live in a table notebook 1 writes and were never in that frame either. |
| **Fix** | **Code, done.** The cell takes the intersection and PRINTS what it skipped, because a characteristic silently missing from a correlation scan is the failure the crash would otherwise have hidden. |
| **What is NOT fixed, and is left deliberately** | `corpus.METRIC_COLUMNS` still omits `modality_index_fitted`, so notebook 3's exploratory correlation scan does not see the modality measure the paper should report. Adding it would change the shape of `TABLE_SyntheticECCMetricsAndW1.xlsx`, which is a regression fixture, and would need notebook 2 rerun and the fixture re-frozen. Nothing depends on it: the metric reduction that uses that measure lives in notebook 4 and reads the corpus directly. |
| **Status** | Crash RESOLVED in Stage 2g; the missing column is open and owned by whichever stage next reruns notebook 2. |

## New, found in the Stage 2g review

## 139. The chance of being the largest contributor is the worst-recovered metric for EVERY method

| | |
|---|---|
| **What was asked** | Whether the finding that the rank metric recovers worst holds across all six UQ methods, or is an artifact of one. |
| **It holds for all six, without exception** | Recovery error on the chance of being largest, and the next worst metric for that same method: kernel estimate with equal weights **0.733** against 0.686; with market-share weights **0.743** against 0.699; lognormal with equal weights **0.719** against 0.658; with market-share weights **0.767** against 0.699; normal with equal weights **1.037** against 0.840; with market-share weights **1.074** against 0.879. |
| **Why that matters for how it is written** | It makes the finding a statement about probabilistic LCA rather than about one way of doing it: **whichever method a practitioner uses, the least reliable number it gives them is the chance that a material is the largest contributor.** It also means the demotion does not depend on which method the paper recommends. |
| **Fix** | **Text.** State it as a property of the metric and not of a method. |
| **Status** | Open, text. Decision 143. |

## 140. No method is best for every claim, and how much the choice costs varies forty-fold

| | |
|---|---|
| **What was built** | Seventeen claims across the five kinds of statement -- attribution, magnitude, action, comparison -- each scored for all six methods as an absolute error against the true distributions, with each row normalized within itself and its stakes measured against the size of the thing being claimed. |
| **The result** | **The strongest single method is best on 6 of the 16 claims where the six differ at all, and four of the six are best on something.** The lognormal with equal weights takes 6, the lognormal with market-share weights 5, the kernel estimate with market-share weights 4, the kernel estimate with equal weights 1. Neither normal is ever first, and both are 5th or 6th on almost every row. |
| **The stakes, which is the part a reader acts on** | How far apart the best and worst methods are, against the size of the claim: **35.5 percent** on a material's chance of being largest, 31.0 on how often a specification cap applies, 30.6 on the cap's chance of delivering 5 percent, 22.1 on a material's share of the total, 20.3 on its contribution -- down to 1.1 on the building total's 90th percentile, **0.8** on whether one design beats another, and **nothing at all** on what a quantity reduction saves. |
| **Fix** | **Text and a figure.** The paper should carry this as a table or a heatmap rather than a single verdict, because a reader's question is "which method for the statement I am making", not "which method". |
| **Status** | Open, text and figure. Decision 143 and the scorecard table. |

## 141. Two of the seven win-share leaders are ties, and an earlier draft named them anyway

| | |
|---|---|
| **What was wrong** | The first version of this stage reported that "three different methods lead" across the seven per-material metrics. It took the argmax of each win share without consulting the bootstrap interval already attached to it. |
| **Measured** | Five of the seven have a leader whose interval clears the runner-up's: the kernel estimate with market-share weights on the chance of being largest (0.221), the 95th percentile (0.258) and the spread (0.296); the lognormal with market-share weights on the estimated contribution (0.236) and the mean share (0.215). **On the uncertainty index the top two are 0.2091 [0.1995, 0.2187] and 0.2082 [0.1993, 0.2177]** -- a margin of 0.0009 on intervals about 0.019 wide -- and on the share at the building's 95th percentile three methods are tied. |
| **So the corrected claim is TWO methods**, not three, and on two metrics the honest answer is that nothing separates. |
| **Fix** | **Text and figure, done in the figure.** Do not name a best method without showing the interval. |
| **Status** | RESOLVED in the analysis; the text must not repeat "three". |

## 142. Only the two lognormals get the specification cap's applicability right

| | |
|---|---|
| **What became askable** | Once the cap's applicability was reported rather than divided away by a constant (entry 134), it could be scored against the truth. |
| **The result** | The true distributions say a cap helps in **0.2786** of iterations. The lognormal with equal weights says 0.2773 and with market-share weights 0.2815 -- **both indistinguishable from the truth**, their bootstrap intervals straddling zero error. The kernel estimate is high by 0.0118 and 0.0200. **The normal is high by 0.0877, which is 31.5 percent too often.** |
| **Why the ordering differs from the rank metric's** | The lognormal's strength is the shape of the upper tail, which is what a cap acts on; the kernel estimate's is following the body of the data, which is what a contribution and its spread are made of. The two orderings are not in conflict; they are about different parts of the distribution. |
| **Fix** | **Text.** Where the paper discusses specification caps, the lognormal is the accurate method and the normal overstates how often the intervention applies by nearly a third. |
| **Status** | Open, text. Decision 147. |

## 143. Uniform, triangular and beta: where they belong, and where they do not

| | |
|---|---|
| **What was asked** | Whether other distributions are worth comparing, and what else is common in LCA. |
| **What is common** | The lognormal is dominant -- it is ecoinvent's default and it is what the pedigree matrix produces, since a geometric standard deviation IS a lognormal parameterization. The normal is common and usually wrong for a strictly positive right-skewed quantity. Both are in the study. Uniform and triangular are used; gamma, Weibull and beta appear occasionally, beta for bounded quantities such as efficiencies, which an embodied carbon coefficient is not. |
| **Why uniform and triangular are not competitors in this comparison** | This paper compares ways of turning a SET of declarations into a distribution. A uniform is not fitted to a dataset: its maximum likelihood fit to n values is exactly the smallest and largest of them, discarding everything between. It would lose by a distance, and **that is the reason to leave it out** -- a family that cannot use the data is a straw man, and a straw man that flatters this paper's own method is worse than no comparison. |
| **Where they do belong** | The judgment-driven arm, with the pedigree matrix, because they are what a practitioner reaches for when there is no dataset. The yardstick that arm uses -- how far apart two models must be before the answer changes -- does not care how either was built. |
| **Gamma is settled and Weibull is scheduled** | Out of sample on the real categories the three-parameter lognormal is indistinguishable from gamma; against the known parent it separates by +0.0117 and +0.0045, winning 77.4 and 67.5 percent of datasets, so it is never worse. |
| **Fix** | **Text**, one paragraph in the methods explaining which families are compared and why these three are not among them. The sweep is Stage 2h's. |
| **Status** | Open, text. Decision 151. |

## 144. The runaway tail exists in this study's own fits, and truncation is the easy fix the paper names

| | |
|---|---|
| **What was asked** | Whether bad tails make much difference here, and whether truncation is worth implementing or worth naming as a weakness with an easy fix. |
| **It is rare and real** | The fitted model's own spread over the data's is near 1.0 in the median for all six methods on both arms, and **0.25 percent of fits exceed five times the data's spread**. The worst reach **73.7**, 50.1 and 40.6 times, and all three are EQUAL-WEIGHTED fits to small datasets; the market-share-weighted twins top out at 5.0, 1.4 and 1.0. |
| **Two guards already catch it** | The bound on the lognormal's threshold at fitting time, and the tail term in the criterion. The part of the score lying beyond the grid is **exactly zero** for the kernel estimate and the normal, which put no mass there, and averages 0.00006 and 0.00009 for the two lognormals. |
| **Why truncation is not implemented** | Every model here is already truncated below at zero, and that bound is external and needs no argument. An upper bound has no equally external anchor: the physical ceiling this study applies to raw declarations is in their own units and every dataset is rescaled to an average of 1.0. Choosing a multiple is a modeling decision with numbers attached. |
| **Fix** | **Text**, one sentence: a distance between cumulative curves cannot see how far out a model puts its rare values, this study charges for it with a tail term and watches it with the fitted-model spread ratio, and truncating each model at a plausible multiple of the largest observed value would remove the failure mode outright at the cost of one more assumption. **The sweep is Stage 2h's.** |
| **Status** | Open, text. Decision 152. |

## 145. CORRECTS an earlier stage: the uncertainty index is NOT "reported nowhere"

| | |
|---|---|
| **What was claimed** | The claim originates in **Stage 2d** -- decision 103 and entry 95, "computed in the pLCA loop as `ui` and appears in no table, no figure and no section of the manuscript" -- and was repeated by Stage 2e in decision 114 and its handoff, and by Stage 2g before checking. All four instances are now annotated in place. |
| **What the manuscript actually does** | Reports it in **Figure 5b and Figure 5d**; defines it in **Supplement 3(c)**, including the formula; discusses it in the results -- "The NRMSE for the uncertainty index is much smaller than that for the ECI Rank #1 Frequency, indicating that different UQ methods result in similar uncertainty indices" -- and again in the conclusions: "pLCA results related to the variance of total embodied carbon, such as the uncertainty index, did not differ substantially between UQ methods." |
| **What IS true, and is the thing worth acting on** | It is reported as a secondary observation about how far apart the methods are, not as one of the questions a probabilistic LCA answers. **The recommendation is to PROMOTE it** to one of the five headline categories -- "which material drives the uncertainty in the total" -- with the finding Stage 2g adds beside it: the methods agree about it to within 4.4 percent of each other and every one of them is out by about half the metric's own between-material spread, and its argmax reading names the truth's answer 58.4 percent of the time against a one-in-four chance level, the best of the seven candidates. |
| **Fix** | **Text.** Do not write "reported nowhere" anywhere. Promote the index from a sentence about NRMSE to a results subsection. |
| **Status** | Correction recorded. Decision 146, correcting decision 114. |

## 146. The five questions a probabilistic LCA answers, as the organizing frame

| | |
|---|---|
| **The frame** | A probabilistic LCA answers five questions, and every result this study reports belongs to one of them: **what is the building's total embodied carbon** (magnitude); **which materials contribute most to it** (attribution); **which materials contribute most to the UNCERTAINTY in it** (information); **how effective is a reduction strategy** (action); and **is this design better than that one** (comparison). |
| **Why it matters here** | It is the structure of the results section and of the claim scorecard, and it is what makes the demotion of the rank metric legible: "which material is biggest" is one of six numbers inside ONE of the five questions, not the study's subject. It also puts the uncertainty index where it belongs -- as the whole of the third question rather than as a footnote to the second. |
| **What the choice of method costs, by question** | Attribution 35.5 percent between the best and worst method at the top of its range, action 31.0, magnitude 4.9, information 2.2, comparison 0.8. **The two questions a designer acts on most directly -- what will the building be, and is this design better -- are the two the choice of method affects least.** |
| **Fix** | **Text and structure.** Use the five questions as the results section's headings. |
| **Status** | Open, structural. Decision 114 established the five; this is the evidence for using them as the frame. |

## 147. The claim scorecard's percentages were two statistics in one unit, and are now one

| | |
|---|---|
| **The defect** | Seventeen claims were drawn on one colour scale as percentages. Seven of them -- the six attribution claims and the uncertainty index -- divided the mean absolute error by the SPREAD of the true value across materials, a signal-to-noise ratio; the other ten divided it by the true LEVEL, a relative error. The percent sign made the two look like one unit. |
| **Why "just name the denominators" is not enough** | The two are not a fixed multiple of each other, so the seven spread-scaled rows were not comparable with each other either. Level over spread runs from **1.17** on the uncertainty index to **6.57** on a material's share of the total, a factor of 5.6. A third inconsistency sat inside the magnitude block: all four of its rows divided by the true building TOTAL, so the error in the total's standard deviation was a fraction of the total's MEAN. |
| **The fix, applied** | Every row now divides by the mean TRUE LEVEL of the same quantity. Every attribution number has a well-defined level -- 1.0397 for a material's mean contribution, 0.6200 for its standard deviation, 2.0812 for its 95th percentile, 0.2500 for each of the three shares and frequencies and for the uncertainty index -- so nothing is dropped for want of a denominator. |
| **What the paper must say about the definition** | It is the mean absolute error divided by the mean true level: a RATIO OF MEANS, not the ordinary mean absolute percentage error, which is a mean of per-case ratios and is not usable here. The true uncertainty index reaches **-0.000671** and **2,904 of 60,000** materials carry a true value below a hundredth of the mean, so a per-material ratio is unbounded and sometimes signless. |
| **One claim is dropped** | `total_w1`, the Wasserstein distance between the method's building total and the truth's. Its true value is zero by definition so it has no level to be a percentage of. The scorecard is 16 claims. It stays in `TABLE_PLCABuildingSummary.csv`. |
| **Both statistics are kept** | `recovery` (error over spread) in `TABLE_MetricRecovery.csv` ranks CANDIDATE METRICS by whether they can tell two materials apart, which is the right question for it and is what the "which metric should the paper lead with" result rests on. `rel_error` (error over level), new in the same table, compares one CLAIM with another and is what the scorecard draws. They must never appear on one axis. |
| **Fix** | **Figure and text.** Decisions 156 and 157; the last paragraph of 156 is annotated as superseded in place. |
| **Status** | Resolved in the analysis. The manuscript owes the definition in one sentence wherever the scorecard is described. |

## 148. The tail figure is cut; the finding is a paragraph

| | |
|---|---|
| **What was cut** | `CompareUQMethods_FIG_TailBlindSpot`, cell and PNG. It showed that a goodness-of-fit score charges for the mass a model misplaces and not for how far out it puts it. |
| **Why** | It is a stress test rather than an observation. Across 60,000 fits on both arms the mean charge for mass beyond the scoring grid is **0.0000 to 0.0001**, and a fraction of a percent of fits exceed five times the data's own spread, because the profile-likelihood guard already bounds it. An upper truncation removes the failure mode outright and Stage 2h owns it. A limitation with a scheduled fix is a paragraph, not a figure. |
| **What survives, with the numbers the sentence needs** | Scored over the scoring grid alone the criterion is FLAT past the grid's top at **9.9 times the dataset mean** -- the same value, **0.262571**, at all 16 contamination distances beyond it, against an uncontaminated 0.254474 -- while over the same range a material's standard deviation moves by a factor of **517**, its mean contribution by 2.0 and the uncertainty index by 1.5. With the tail term Stage 2c added the score climbs to **3.27** instead of staying flat. Shares and rank frequencies saturate because they are bounded in [0, 1]. |
| **Where it lives** | `TABLE_MetricTailStress.csv` and `TABLE_MetricTailReality.csv`, printed by two notebook cells. |
| **Fix** | **Text.** One paragraph in the limitations, pointing at the truncation sweep. Decision 158; decision 149's numbers are unchanged. |
| **Status** | Resolved in the analysis. |

## 149. The uncertainty index is the most consistent output AND the worst recovered, and the reason is dataset size

| | |
|---|---|
| **The apparent contradiction** | The manuscript concludes that "different UQ methods result in similar uncertainty indices", and that is right: the NRMSE between methods is 0.5035 [0.4918, 0.5158], the lowest of any output. Against the TRUE parent every method is out by about **43.6 to 45.5 percent** of the true level, the worst of the sixteen claims. |
| **Why both are true** | Decomposing each method's per-material error into the part all six share and the residual: the shared part is **0.1017** of a total error of about **0.11**, so **nine tenths of the error is common to every method**. The six errors correlate 0.66 to 0.99 and all six err in the same direction on **56.9 percent** of materials against about 3 percent if independent. |
| **The mechanism** | Error by the material's own dataset size: **0.166** at n = 3-9 with a signed error of **-0.093**, 0.112 at 10-99, 0.084 at 100-999, and **0.073** at n >= 1000 with a signed error of **+0.046**. Every method understates the variance of a material estimated from three to nine values, and because the index is a variance SHARE summing to one, the share the small material loses is handed to the large ones. It is a property of the data, not of the method. |
| **And the true value is not a stable 0.25** | Across 10,000 materials the true index runs from **0.009 at the 10th percentile to 0.558 at the 90th**, standard deviation 0.214, maximum 0.984. So 43.6 percent of 0.25 is an absolute error of 0.109 on a quantity that genuinely spans almost nothing to almost everything. |
| **Fix** | **Text.** Both halves in one sentence wherever the index is promoted: the choice of method barely matters for it, and every method answers it worst. Decision 159. |
| **Status** | Open, and it sharpens rather than contradicts the existing conclusion. |

## 150. "Variable weighting" names a claim the method does not make

| | |
|---|---|
| **The defect** | "Variable" reads as "market shares accounted for". It means the shares were drawn from a flat Dirichlet because nobody publishes them. Read the first way, a result where equal weighting beats it looks like a modeling error, which is how the author read it. |
| **What the oracle run settles** | Mean absolute error in a material's estimated contribution against the market-weighted truth, over 1,200 pLCA groups -- equal weights, Dirichlet-drawn shares, then the TRUE shares: KDE **0.1285 / 0.1215 / 0.1064**, lognormal **0.1243 / 0.1168 / 0.1007**, normal 0.1620 / 0.1629 / 0.1550. On a material's chance of being largest: KDE 0.0815 / 0.0818 / **0.0717**, lognormal 0.0798 / 0.0847 / **0.0731**. |
| **The claim the paper can make** | **Knowing the market shares beats equal weighting on every metric and every family.** Guessing them with a flat Dirichlet captures about a third of that on the magnitude and less than nothing on the ranking -- the lognormal's ranking error goes 0.0798 to 0.0847 when guessed and would have gone to 0.0731 had they been known. The mechanism is the effective sample size: a flat Dirichlet over n points leaves a Kish effective sample of about n/2. |
| **And the scorecard differences are real** | Paired cluster bootstrap over pLCA groups, equal minus Dirichlet-drawn, negative meaning equal weighting is closer: chance of being largest **-2.14 [-3.12, -1.35]**, share of the total -0.62 [-0.91, -0.29], the total's standard deviation -1.62 [-2.27, -0.95], mean contribution **+0.61 [+0.35, +0.91]**, 95th percentile +1.34 [+0.96, +1.73]. Guessed shares help the levels and hurt the ranking. |
| **Fix** | **Figures, tables and text.** Display label becomes "Dirichlet shares"; the stored column keeps "Variable" as the join key. "Dirichlet" rather than "guessed" because it is the instrument Torres et al. (2026) puts in its own title, so the two papers stay consistent. Decision 160. |
| **Status** | Applied to this stage's figure. **Stage 3 owns the remaining figures.** The manuscript owes the same change wherever it names the scheme. |

## 151. The claim scorecard's box count is not a ranking of methods

| | |
|---|---|
| **The risk** | Pooled over every dataset size the scorecard gives the lognormal 10 best-method rows to the kernel estimate's 5, which reads as a verdict for the lognormal. |
| **Why it is not one** | Mean error across the seven per-material claims, by the material's own dataset size, as a pct of each claim's true level: at n = 3-9 the equal-weighted lognormal leads at 40.1, at 10-99 it leads at 23.7, at 100-999 the Dirichlet lognormal leads at 15.3, and at n >= 1000 the **Dirichlet kernel estimate** leads at **11.0** against the equal-weighted lognormal's 17.2. Both axes invert: equal weights win every band below 100 declarations and Dirichlet shares win every band above. |
| **Where the pooled count comes from** | The corpus allocates 2,500 datasets to each of four size bands, so half of every pLCA sits below 100 declarations. That allocation is an experimental design choice, not a claim about the world. |
| **And the real world does not rescue it** | Reweighting to the real size mix of the 147 EC3 categories -- 14 / 54 / 26 / 6 percent -- moves the count FURTHER toward the lognormal, because two thirds of real categories hold fewer than 100 declarations. |
| **The narrative the paper can defend** | The normal is the one clear loser, and even that is conditional: at three to nine declarations nothing can be estimated and a normal is as good as anything. Kernel estimate versus lognormal is a size rule at about 81 declarations, not a verdict. Whether to weight is a size rule too. |
| **Fix** | **Figure and text.** The scorecard carries a second panel of four size bands by six methods on the same colour scale. A caption warning was rejected: it would ask the reader to take the caveat on trust rather than showing the mechanism. Decision 161. |
| **Status** | Resolved in the analysis. The manuscript must not quote the pooled count as a ranking. |

## 152. A threshold measured on one dataset's fit does not transfer to a group's answer

| | |
|---|---|
| **The conflation** | The study's practitioner rule -- use a kernel estimate above about **81** declarations and a three-parameter lognormal below -- is measured on W1 against the parent for a single dataset. It has been quoted as though it also described when a probabilistic LCA's ANSWERS get better, and it does not. |
| **A better fit DOES give a better answer, which had to be established first** | Holding the material fixed and ranking the six methods by fit and by claim error: median within-material Spearman **+0.600**, positive on **82.2 percent** of materials, and the best-fitting method is also the most claim-accurate on **39.0 percent** against a 16.7 percent chance level. There is no finding that fit fails to translate. |
| **What differs is the UNIT, not the relationship** | All four materials in a pLCA are fitted by the same method, so the group decides. Binned by the FOCAL material's n the kernel estimate lags through 82-200 (19.8 vs 18.2) and leads above 500; binned by the GROUP's MEDIAN n it leads in **seven of eight bins**. Two partitions of the same 10,000 materials. **"500" is therefore not a second threshold and must not be quoted as one.** |
| **The group dilution, measured** | Materials with n > 1000 whose group's smallest dataset is also above 1,000 sit at **3.6 against the lognormal's 6.9**; with a 3-9 material in the group it is 13.9 vs 15.7. That configuration occurs **48 times in 10,000**, because groups are random. |
| **And the advantage is in the spread, not the level** | At n > 1000 the kernel estimate's margin is +6.1 points on a material's standard deviation, +2.2 on its chance of being largest, +1.8 on the uncertainty index, +1.4 on its 95th percentile -- and it ties or slightly loses on the mean contribution and the mean share. |
| **Fix** | **Text.** Every threshold this study quotes must name the criterion AND the unit it was measured on -- one dataset's fit, or one group's answer. Decision 163, narrowed the day it was written. |
| **Status** | Open. It narrows how decisions 139 and 142 may be quoted; it does not change what they measured. |

## 153. The manuscript must compare its lognormal with the TWO-parameter one, in the results

| | |
|---|---|
| **Why** | The study's lognormal is a THREE-parameter fit with the threshold chosen by profile likelihood. The field's lognormal is the TWO-parameter one: ecoinvent's default, and what the pedigree matrix produces, since a geometric standard deviation is a lognormal parameterization. A reader will assume they are the same and conclude the paper rediscovered current practice. |
| **The machinery the paper had to build** | A three-parameter lognormal has no global MLE -- the likelihood is unbounded as the threshold approaches the smallest observation. So: `fit_lognorm3_profile`, which chooses the threshold by profile likelihood; the guard `PROFILE_DELTA_LO_FRAC = 0.25`, calibrated on a bounded-variance criterion and NOT on W1, which sets the threshold for **48 percent of empirical fits**; explicit truncation to (0, inf) with renormalization; and the analytic tail term. At a guard of 0.01 the same estimator produced a model with a standard deviation of **3,281** on data whose own is 0.6. |
| **The numbers** | Against the known parent, 1,500 corpus datasets, equal weights. Closest on: **kernel estimate 47.8 pct**, three-parameter lognormal 17.8, two-parameter lognormal 14.7, normal 10.5, gamma 9.3. Median relative gain over the two-parameter lognormal: the three-parameter lognormal **-11.5 pct** at n = 100-999 and **-14.8** at n >= 1000; the **kernel estimate -30.8 and -41.2**. The kernel estimate is closer than the two-parameter lognormal on **71.5 percent** of datasets. |
| **Fix** | **Text and a table.** Put the two-parameter comparison in the results. Without it the paper's headline reads as "use a lognormal", which is what readers already do and is not what the evidence says. `audits/lognormal_variants.py`, decision 167. |
| **Status** | Open, and it is the single most important framing change the manuscript needs. |

## 154. Goodness-of-fit DOES predict pLCA accuracy; do not write that it does not

| | |
|---|---|
| **The tempting wrong claim** | That fit is irrelevant to probabilistic LCA outcomes, because the kernel estimate overtakes the lognormal on fit at about 81 declarations and does not sweep the downstream claims there. |
| **What is measured** | Holding the material fixed and ranking the six methods by fit and by claim error: median within-material Spearman **+0.600**, positive on **82.2 percent** of materials, and the best-fitting method is also the most claim-accurate on **39.0 percent** against a 16.7 percent chance level. |
| **Why the kernel estimate does not sweep above 81** | Because it does not sweep the FIT above 81 either. It wins 61.2 / 53.0 percent of datasets at n = 82-200 by a median of 2 to 5 percent, and 86.5 / 95.7 percent at n > 3000 by 22 to 68 percent. 81 is where it crosses half, not where it dominates. The claim-level picture tracks that faithfully. |
| **Fix** | **Text.** State the size gradient, not a threshold, and never state that fit does not matter. Decision 166. |
| **Status** | Open. |

## 155. The corpus's multimodal datasets are the wrong shape, and the reassurance on record does not cover it

| | |
|---|---|
| **The defect** | Spearman of the visible mode count with each characteristic has the **opposite sign on all six** between the arms: real multimodality comes with more spread, skew and kurtosis (+0.163, +0.211, +0.230), the corpus's with less (-0.153, -0.152, -0.144). A three-mode real category has a coefficient of variation of 0.974; a three-mode synthetic one has **0.230**, tighter than the corpus's own unimodal median. |
| **The mechanism** | The generator makes a visible mode by SEPARATING components -- median achieved overlap 0.513, 0.422, 0.411 as modes go 1, 2, 3 -- and separated components are individually tidy. Real multimodality is a shoulder on a long-tailed body. Decision 37 predicted this in Stage 2a-2 and it was never measured against the mode count. The calibration objective matches marginals one at a time and never looks at a correlation between characteristics, so every margin can match while the joint is backwards. |
| **Why the existing reassurance does not apply** | Decision 82 reweighted the corpus to the empirical mode mix and found the kernel-minus-lognormal difference moved 0.0004. **Reweighting can only reweight datasets that exist**; it cannot create the population the corpus lacks. |
| **The size of it** | Multimodal AND dispersed is **1.9 percent of the corpus against 16.2 percent of the real arm**, an 8.4-fold under-representation covering a sixth of real categories. |
| **What it does NOT show** | That the corpus is biased against the kernel estimate. On the corpus's own multimodal-and-dispersed datasets the kernel estimate does worse, and on the real arm's 21 such categories it wins 47.6 / 38.1 percent. The defensible statement is that **the corpus cannot speak to a sixth of real categories**, not that the answer would change. |
| **Fix** | **A limitation now; possibly a regeneration.** Stated as it stands, the corpus spans the dispersion and the modality of real categories separately and not jointly. Decision 168 records the diagnosis, the proposed generator change and that the decision is the author's. |
| **Status** | Open, author decision. Generation remains closed until then. |

## 156. The corpus spans modality and dispersion separately, not jointly, and the fix costs the weighting result

| | |
|---|---|
| **The limitation, stated as it should appear** | The synthetic corpus reproduces the distribution of visible modality across real ECC categories and the distribution of dispersion across them, and does NOT reproduce their joint distribution. **Conditional on being dispersed, a real category is multimodal 44.7 percent of the time and a synthetic one 9.4.** Multimodal AND dispersed is **1.9 percent of the corpus against 16.2 percent of the real arm.** |
| **The mechanism** | Dispersion in the generator is produced by the SHIFT -- Spearman(shift, achieved CV) = **-0.654**, against **-0.017** for the achieved overlap. The smallest admissible shift is `min_q1_over_iqr * (q3 - q1) - q1`, so a wider mixture is forced to shift more, which caps its coefficient of variation lower. Separating components therefore COSTS dispersion, and 94.8 percent of the most-dispersed quartile is already pinned against that floor. |
| **It was fixed, and the fix was rejected** | `genconfig.separation_dispersion_frac` solves the component SPACING for the coefficient of variation and leaves the shift at the floor, so dispersion arrives with separation. On a 1,000-dataset draft it takes multimodal-and-dispersed from **1.27 to 12.66 percent** and the multimodal share to 0.330 against a real 0.315. It also makes the corpus OVERSTATE the paper's headline effect: the median uniform-to-variable Wasserstein distance goes from **0.1035 against a real 0.0929** to **0.2513, 2.7 times the real median**, with a maximum of 13.08 against a real 0.73. The standardized arm-to-arm distance on that characteristic goes from 0.2915 to 2.5732. **No value of that quantity is intrinsically better or worse -- the objection is calibration**: the corpus would say weighting matters 2.7 times more than real categories say it does. |
| **Why that settles it** | A corpus that matched modality and dispersion while misrepresenting the weighting effect by an order of magnitude would be a worse instrument for this paper than the one that exists. The same trade has now appeared three times on three different levers (decisions 39, 138, 170), so it is structural: whatever widens dispersion in this generator also widens the gap between the two weightings. |
| **What was searched** | 36 configurations -- 13 draft corpora and 23 fast probes. The modality-shape correlation is negative in every one; the best is -0.083 on the coefficient of variation against a real +0.163. |
| **Fix** | **Text, in the limitations.** State the joint gap and what it bounds: the study cannot speak to the sixth of real categories that are both multimodal and dispersed. Do not claim the corpus spans the space of real ECC datasets; claim it spans each margin. Decisions 168, 169, 170. |
| **Status** | Open for the manuscript. The code is committed, defaulted off and tested, so a later stage can revisit it -- but only together with the weight model, which is Stage 2h's first item, because the two are coupled. |

## 157. The design-comparison null is about an average, not about one comparison

| | |
|---|---|
| **What the paper currently says** | That the choice of UQ method changes the stated probability that a substitution is an improvement by at most **0.015**, and every method lands within 0.026 of the truth. Decision 118. |
| **What that number is** | The spread of the AVERAGE probability over 2,500 design pairs. It is not the error on any one comparison. |
| **On a single comparison** | The six methods span a median of **0.182** in probability when B claims no saving, **0.165** at a claimed 5 percent saving, **0.119** at 10 percent and 0.025 at 20. The median per-comparison absolute error against the truth runs **0.039 for the best method and 0.079 for the worst**. |
| **And they disagree about the answer** | Share of individual comparisons on which at least two of the six land on opposite sides of 0.5 -- that is, disagree about which design is better: **93 pct** at a claimed 0 pct saving (the control: the truth is a coin flip there, so this means nothing), 80 at 1 pct, **64 at 2 pct**, **28 at 5 pct**, 5.9 at 10 pct, 0.04 at 20 pct. |
| **The honest statement** | The null holds where the design difference is real and fails where it is small. A practice that makes many comparisons is safe under any of these methods; a designer comparing two options that are within a few percent of each other is not. This is the same near-tie fragility the paper already records for the ranking metrics, arriving at the design question. |
| **A related defect in the scorecard** | Five of its sixteen rows -- the four reduction-strategy rows and the design comparison -- are computed from summary tables whose errors were averaged over groups before the absolute value, while the other eleven take the absolute value per group or per material. The most misleading is "using 25 percent less", which reads as every method being EXACTLY right when on a single building every method is 10 to 13 percent out. |
| **Fix** | **Text, and a decision about the figure.** Either make all sixteen rows per-unit, which moves the figure's headline from "right to 0.8 pct" to "right to 12.0 pct", or keep the averaged form and say on the figure that those five rows are errors in an average. Decision 171. |
| **Status** | Open, author's call. Nothing is wrong with the underlying run. |

## 158. The scorecard's five averaged rows are corrected, and two published sentences change

| | |
|---|---|
| **What changed** | Entry 157 left this as the author's call and the author took the correction. All sixteen scorecard rows are now the mean absolute error PER UNIT over the claim's own true level. |
| **The five rows, best method, as a percentage of the true level** | how often a cap binds **0.48 -> 30.62**; a cap's mean saving **0.56 -> 38.29**; a cap's chance of saving 5 percent **1.09 -> 32.97**; what using 25 percent less saves **0.00 -> 10.02**; the probability B beats A **0.81 -> 11.95**. The other eleven rows are bit-identical. |
| **It is a change of definition and that is checked** | The new `portfolio_error` column reproduces the old values to 2e-15. Both statistics are kept and labelled: `total_error` is the error in a SINGLE decision, `portfolio_error` the error in the AVERAGE claim over many. |
| **SENTENCE ONE THAT MUST CHANGE** | "On what using 25 percent less of a material saves, every method is exactly right, because that intervention is a deterministic fraction of the material's own contribution and no distributional assumption enters." **False for one building**: every method is 10.0 to 13.4 percent out per pLCA. It is true of the AVERAGE over many, and the paper must say which. |
| **SENTENCE TWO THAT MUST CHANGE** | "On how often a specification cap binds the choice costs 31.0 and the best method is 0.5 out: picking well is nearly the whole problem." **Reversed.** Per decision the best method is 30.6 percent out and the choice costs 23.1, so most of the error is there whatever is chosen. |
| **And two counts** | The six differ measurably on **16 of 16** claims rather than 15, because the quantity-reduction row now differs. The figure's headline moves from "right to 0.8 percent on the design comparison" to **"right to 12.0 percent"**. |
| **What does NOT change** | No normal fit is best on any of the sixteen claims and it is the worst on thirteen. The most expensive QUESTION is still `action`; the claim inside it moves from "how often a cap binds" to "a cap's chance of saving 5 percent", at 25.1 percent. |
| **Fix** | **Text.** Every quoted scorecard number for those five rows, plus the two sentences above. Decision 174. |
| **Status** | RESOLVED in the analysis. The text edits are open. |

## 159. Every published crossing needs both fits and fewer printed digits

| | |
|---|---|
| **The defect** | The intervals on the flip thresholds and the safe-lead ratios are bootstrap intervals on a fitted LOGISTIC's parameters. They say how well the data pin down that curve and nothing about whether a logistic is the right shape. |
| **Measured** | An isotonic fit -- which assumes only that the probability does not fall as the models separate -- falls OUTSIDE the logistic interval on **five of the six** published constants. The flip threshold at 1 percent: logistic 0.00175, interval [0.00131, 0.00227], isotonic 0.00259. The safe lead at 10 percent: 1.46019, [1.44678, 1.47196], isotonic 1.35217. |
| **Not an extrapolation, though** | All six crossings are bracketed by binned observations on either side, over 22,500 calibration rows and 72,000 four-material comparisons. Nothing needs re-deriving. |
| **What prose may print** | The significant figures the two fits agree on, plus the first they part at, and BOTH values where they still differ there: safe lead 1 pct **2.1 against 2.2**, 5 pct **1.64 against 1.61**, 10 pct **1.5 against 1.4**; flip 1 pct 0.002 against 0.003, 5 pct 0.011 against 0.013, 10 pct 0.02 against 0.03. |
| **What the tables must carry** | Both fits and the interval at FULL precision, in the results tables and the supplement, because a reader checking the work or carrying a constant downstream needs the unrounded value. Five columns were added and none edited. |
| **Fix** | **Text.** Round every crossing stated as a rule in prose or a figure annotation; leave the tables alone. Decision 175. |
| **Status** | RESOLVED in the analysis. The text edits are open. |

## 160. The two arms weighted their data by different rules. THE FIX IS NOW APPLIED and reported numbers have moved

| | |
|---|---|
| **The defect** | The real categories drew market shares from a flat Dirichlet over individual declarations, so shares were INDEPENDENT of the carbon coefficients; the synthetic datasets attached a share to each mixture component and split it inside, so shares were CORRELATED with them. Independent weights are exchangeable, so the measured weighting effect MUST decay like n^-1/2 whatever markets do. |
| **Measured, median separation by size band and the decay slope** | empirical as it stands 0.2138 / 0.1189 / 0.0678 / 0.0050, slope **-0.449**; synthetic as it stands 0.1574 / 0.1260 / 0.0595 / 0.0528, slope **-0.181**; the synthetic arm's own values reweighted by the empirical rule 0.1330 / 0.0938 / 0.0375 / 0.0124, slope **-0.367**. The third row is the proof that the gap is the RULE and not the data. |
| **The fix, validated where the truth is known** | Cut the sorted declarations into contiguous groups, draw each group's share, split inside it. On the synthetic arm, where the true mode labels exist, a contiguous cut at coherence **rho = 0.5** reproduces the true-label weighting effect to within 4 percent on the typical dataset and orders the categories most like the truth (Spearman 0.935). rho = 0 gives 0.55 of the true effect and rho = 1 gives 1.60. |
| **And the concentration anchors independently** | Drawing the group count the way the generator draws its component count -- uniform on 1 to 5, independent of n -- gives a median top-ROUTE share of **0.6267**, against the published **0.6375** for Rest-of-World BOF steel and 0.54 for China's share of global production. |
| **THE RESIDUAL ARM GAP IS DISPERSION, NOT THE RULE** | Under one rule, dividing each dataset's separation by its own coefficient of variation, the arms agree in every size band at every coherence: at rho = 0.5, empirical 0.3964 / 0.2422 / 0.1547 / 0.1579 against synthetic 0.4021 / 0.2577 / 0.1668 / 0.1420. So what is left is the size-and-dispersion law holding on both arms with different dispersion fed in, which is entry 156's shortfall. |
| **rho = 0 is not the neutral choice** | It is the claim that market share is uncorrelated with carbon intensity, and published production volumes contradict it: 63.75 percent of world steel is on the higher-carbon route and 0.03 percent on Austrian EAF. |
| **APPLIED 2026-09-25, by author decision** | `empirical.WEIGHT_RHO = 0.5`. All twelve UNWEIGHTED characteristic columns are bit-identical, which is the control; all twelve weighted ones move, by a median absolute 0.066 on `coeffvar`, 0.066 on `w_v_uw_wasserstein`, 0.135 on `entropy` and 0.648 on `skewness`. All six W1 scores move, correctly, because every model is scored against the variable-weighted empirical CDF. Arm medians: `w_v_uw_wasserstein` **0.1048 to 0.1475**, `coeffvar` 0.6706 to 0.6406. Three regression fixtures re-frozen. |
| **What it closes** | Decay on log10(n) goes from -0.449 to **-0.161** on the real arm against -0.101 on the synthetic under the same rule, and the median effect above a thousand declarations from 0.0050 against 0.0528 to **0.0441 against 0.0848**. The tenfold arm disagreement is now a factor of 1.9. |
| **An unarranged side effect** | The arm-to-arm distance on `fit_lognorm_SF`, which decision 37 called the worst characteristic in the project and structural after four failed attempts, **halves from 0.65 to 0.33**. Nothing was tuned for it. |
| **Fix** | The manuscript must state that the arms previously weighted differently, that the published claim "weighting stops mattering above about a thousand declarations" was largely an artifact of that, and that under one rule the effect up there is about nine times what the paper reports. Every weighting number in the paper is recomputed from the new run. Decisions 178, 179, 190. |
| **Status** | **APPLIED.** Reported numbers moved; see decision 190 for the full list. |

## 161. A single weight realization is not the distribution, and one published R2 depends on which

| | |
|---|---|
| **The two quantities** | A per-dataset weighted statistic is ONE draw from a distribution; an arm-level statistic is a property of that distribution estimated from 147 categories. |
| **Measured over 25 independent realizations of the whole arm** | Per dataset, `w_v_uw_wasserstein` has a typical standard deviation of **0.0459 on a median of 0.1076**, which is 47 percent relative, and a worst range of 1.37; kurtosis reaches a worst range of 904. Arm level, the median of the same statistic is **0.0981 +/- 0.0052**, five percent relative. A factor of nine. |
| **The control** | Every unweighted twin moves EXACTLY zero, which is what says the measurement is the weight draw and nothing else. |
| **THE PUBLISHED FIGURE THAT DEPENDS ON THIS** | The size-and-dispersion law is reported with an R2 of **0.991**, and that is on the MEDIAN separation over 1,000 draws. On a single realization the same law explains **0.824 +/- 0.029**, and the Spearman with dispersion is **0.573** against the published 0.731. The stored characteristic IS a single realization, so a reader recomputing the law from the published characteristic table will not reproduce 0.991. |
| **Fix** | **Text.** State which of the two any quoted R2 or correlation came from, and state per-dataset weighted claims as distributions rather than numbers. The law's exponents are safe either way: -0.448 +/- 0.023 and 0.897 +/- 0.044. Decision 180. |
| **Status** | Open. |

## 162. Three items the stage was told to check and could close

| | |
|---|---|
| **The industry-average declaration** | EC3 does carry a `declaration_type` field and the frozen extract does record it. **All 120,280 records read `Product EPD`.** There is no industry-average declaration to compare a category's uniform mean against, so the check is dropped rather than inferred from product names. The idea remains the right instrument for the question and is not testable on this data. Decision 176. |
| **Six-or-more-mode datasets** | There are none. At the bandwidth the study fits, the corpus runs 0.760 / 0.212 / 0.024 / 0.004 / 0.001 over one to five modes with a MAXIMUM of five, and the real categories 0.685 / 0.262 / 0.054 with a maximum of three. The "5.6 percent against 0.7 percent" on record is a Silverman critical-bandwidth count on a corpus regenerated twice since. What survives is a mild over-representation at four and five modes, 0.5 percent of the corpus against nothing real. Decision 177. |
| **The joint modality-dispersion cell, re-measured** | On one definition for both arms -- two or more visible modes at the fitted bandwidth, coefficient of variation at or above 1.0 -- the corpus reaches 0.240 multimodal against a real 0.315, which is close, and 0.058 dispersed against a real 0.269, which is a factor of 4.6. **So the joint gap is mostly the dispersion marginal of entry 156 rather than a separate defect.** Excising the 97 both-at-once datasets moves the share on which the kernel estimate beats the lognormal from 0.6545 to 0.6571. The cell itself reads the other way, 0.392, so the gap does not hide a kernel-estimate win. |
| **Fix** | **Text.** Drop the industry-average plan, drop the six-mode caveat, and state the joint gap as the dispersion shortfall it mostly is. |
| **Status** | RESOLVED. |

## 163. Splitting the categories did not manufacture the headline

| | |
|---|---|
| **The question** | Stage 2a-3 resolved the EC3 categories into specifiable products by three metadata rules. The evidence that this did not produce the result is the UNSPLIT arm run against every headline. |
| **The three arms** | primary 147 datasets and 116,766 values; unsplit 136 and 119,448; deduplicated, one record per (manufacturer, product name), 147 and 65,839. |
| **The win share, cross-validated, is essentially unchanged** | equal-weighted lognormal 0.362 primary against 0.358 unsplit and 0.386 deduplicated; equal-weighted kernel estimate 0.260 / 0.236 / 0.244; equal-weighted normal 0.118 / 0.098 / 0.134. **The method ordering is identical on all three.** |
| **The levels are higher on the unsplit arm and that is expected** | Median cross-validated W1 for the equal-weighted lognormal 0.2613 primary against 0.3006 unsplit, because an unsplit category mixes products and is harder to fit. The characteristics move the same way: median coefficient of variation 0.675 primary against 0.815 unsplit. |
| **The size crossover cannot be measured on this arm at all** | 274 primary, 221 unsplit, 463 deduplicated under equal weights, and 661 / 334 / 3377 under Dirichlet shares. That is the instability decision 136 already records for 127 categories, not a sensitivity to the population definition, and it is a different quantity from the corpus's 81 in any case. |
| **Fix** | **Text.** Report the unsplit arm against the win share and the characteristic medians, and do NOT report a crossover from the empirical arm. Decision 183. |
| **Status** | RESOLVED. |

## 164. The judgment-driven methods, placed on the same axis

| | |
|---|---|
| **What was built** | A pedigree model as a lognormal specified by a centre and a geometric standard deviation, plus a uniform and a triangular over a range derived from the SAME two inputs, so the three differ in shape and not in information. They are NOT in the main comparison, because a family that cannot use the data would be a straw man that flatters this paper's own method. |
| **At the fit level they are far worse** | Against the true distribution a judgment model is 2 to 100 times further away than the best data-driven fit. The realistic model -- the centre drawn as one random declaration, which is what a practitioner without a dataset holds -- is 4.1 to 12 times worse. |
| **At the decision level a well-centred one is competitive** | On the design comparison the six data-driven methods span 0.080 to 0.114 error per pair, and a pedigree model centred on the category mean reads **0.097 at a matched spread and 0.097 to 0.121 across a six-fold range of spread**. The spread axis is nearly flat. |
| **WHAT BREAKS IT IS THE CENTRE, NOT THE SPREAD** | A displacement applied to every material alike cancels EXACTLY, because both design options' totals scale by the same factor. A displacement drawn PER MATERIAL does not: 0.111 at 10 percent of the mean, 0.159 at 25 and 0.223 at 50. The realistic one-declaration model reads 0.170 to 0.271, one and a half to two and a half times the worst data-driven method. |
| **And the shape matters** | At a matched spread and a correct centre the pedigree lognormal reads 0.097 where a uniform reads 0.206 and a triangular 0.198. |
| **The deliverable sentence** | A judgment-driven model with a plausible spread gives the same design answer as a data-driven one PROVIDED its point estimate is not displaced; what a practitioner actually has -- one declaration per material -- displaces it independently for each material, and that roughly doubles the error. |
| **A sourcing gap the manuscript must close** | The pedigree matrix's uncertainty-factor table is not in this repository's reference folder, so the spread was swept RELATIVE to the data's own rather than in absolute pedigree units. **A specific pedigree score cannot be laid on this axis until that table is sourced.** This project has already had to withdraw one figure quoted from memory. |
| **Fix** | **Text.** Report the two-dimensional result, lead with the centre, and source the factor table before quoting a score. Decision 184. |
| **Status** | Open for the manuscript. |


## 165. One scorecard claim is an identity of another

| | |
|---|---|
| **What was found** | After the per-unit correction, "what using 25 percent less of a material saves" and "a material's share of the building total" carry identical numbers in all six cells of the scorecard. |
| **Why** | Using 25 percent less of a material removes exactly a quarter of that material's share of the total, with no distribution entering, so the error in the first is exactly 0.25 times the error in the second and the true levels stand in the same ratio (0.0625 against 0.2500). Verified over all 60,000 rows: largest deviation 1.1e-15, correlation 1.00000000. |
| **The old definition hid it** | Under the averaged form both rows read 0.00, which reads as two independent claims agreeing rather than as one claim counted twice. |
| **Fix** | **Text.** Report one of the two, and state that a quantity reduction is a deterministic fraction of a material's own share so its accuracy IS that share's accuracy. Do not present them as two pieces of evidence. The figure keeps both rows because it is grouped by the five questions a reader asks. Decision 186. |
| **Status** | Open for the manuscript. |

## 166. The comparison margin is a tolerance, and the text describing it says the opposite

| | |
|---|---|
| **The code** | `plca.comparison_statement` computes `P(a < g * b)` where `a` is the proposal and `b` the baseline. With `g` ABOVE one this asks "is the proposal better, OR worse by less than `g`", which is a TOLERANCE. |
| **The text that was wrong** | The docstring read "the share in which A beats B by a margin worth acting on", which describes `g` BELOW one. |
| **Confirmed on the study's own run** | At a true 20 percent saving `mci_1.2` reads **0.9993** against a discernibility of 0.9628. A stricter condition cannot exceed a looser one, so the margin is the loose direction. |
| **What does NOT change** | The reported `mci_1.05` and `mci_1.2` values are correct AS TOLERANCES and are what Marsh et al. (in press) report at 1.2. **No number moves.** |
| **Fix** | **Text.** Wherever the manuscript describes the modified comparison index, say that a margin above 1 admits a proposal that is slightly worse, and do not describe it as a margin of superiority. Decision 187. |
| **Status** | RESOLVED in the code comment and the tests. The manuscript text is open. |

## 167. The certification credit is a decision the paper can speak to, and it is fragile at the bar

| | |
|---|---|
| **The framing** | Certification awards points for demonstrating a reduction against a baseline. Under a probabilistic LCA that becomes "demonstrate a 10 percent reduction with 75 percent confidence". The study computes the probability and never asked whether the method says the credit is EARNED. |
| **The result** | Over 600 design pairs at five true savings: the truth earns the credit in 17.4 percent of cases, at least two of the six methods disagree in **18.2 percent**, the best method is wrong 8.8 percent of the time and the worst 12.5. |
| **And the fragility is the THRESHOLD** | Split by how far the true confidence sits from the line: **65.3 percent** disagreement within 0.05 of it, 46.3 between 0.05 and 0.10, 20.5 between 0.10 and 0.25, and **3.4 percent** beyond 0.25. |
| **Which method is worst** | A normal fit, on 8 of the 15 tier-and-confidence combinations. |
| **The claim the paper can make** | A design comfortably over or under the bar is called the same way by every method; a design sitting on the bar is decided by the modeling choice. That is an argument for stating the margin, not against writing credits probabilistically. |
| **A SOURCING CONSTRAINT** | The tiers used are the study's own 5, 10 and 20 percent. **The exact wording, tier and confidence of any specific credit must be sourced before the paper cites one**, on the same grounds as the withdrawn ICE figure and the pedigree factor table. |
| **Fix** | **Text, a new result.** Decision 187. |
| **Status** | Open for the manuscript. |

## 168. The dispersion-versus-weighting trade was an artifact of the weighting mismatch, and the corpus is no longer blocked

| | |
|---|---|
| **What the manuscript currently owes** | Entries 127, 155 and 156 and decisions 138, 169 and 170 all record the same limitation: the corpus cannot reach the dispersion of the most variable real categories, and every attempt to widen it inflated the median uniform-to-variable Wasserstein distance past the real arm's, which is the paper's headline quantity. 36 configurations on four levers across three stages, and the sign never flipped. It was recorded as structural. |
| **It is not structural** | With both arms on one weight rule (entry 160), bounded widening candidates improve BOTH at once. Objective 0.2216 at the shipped configuration against 0.1706 to 0.1912 for four candidates, which is **4.6 to 7.7 seed standard deviations better**; the standardized `coeffvar` distance halves from 0.4085 to 0.2040 and `w_v_uw_wasserstein` falls from 0.3400 to 0.1528. Absolute and standardized distances agree in sign on every characteristic, so this is not the denominator artifact entry 63 records. |
| **The counterfactual, which is what makes it a mechanism** | One synthetic draw scored against the empirical arm built four ways, so only the empirical weighting differs. Change in the weighting distance from the shipped configuration, in units of its own 0.0351 seed sd: old flat rule **+4.4 and +6.8**; ported at rho = 0 **-1.4 and -0.4**; at rho = 0.25 **-0.5 and +1.8**; at rho = 0.5 **-5.1 and -5.3**. The sign flip is the change of RULE, not the value of rho. |
| **Why** | Porting the rule raised the real arm's median weighting effect from 0.105 to 0.148 while the shipped corpus sits at 0.092, so the corpus now understates it by a third and widening moves toward the arm rather than past it. |
| **The one cost, measured** | `crit_bw_1` worsens from 0.158 to 0.259 standardized, on a characteristic carrying weight 3 in the objective. Reweighting the corpus to match the real arm on it moves the KDE win share by -0.006 and -0.015, against -0.026 and -0.043 for `coeffvar`, so it is worth about a quarter of what is gained, in the same direction. |
| **Fix** | The stated limitation about dispersion is still TRUE of the shipped corpus and must stay until the corpus changes. What must NOT be written is that it is unfixable, or that fixing it costs the weighting result; both were true only under the mismatched rules. Decisions 193, 196. |
| **Status** | Open, and it is an author decision. Nothing is regenerated. Any candidate acted on must first pass the parent-level gate of entry 169. |

## 169. A generator setting can score well and be unusable, and the check that catches it did not exist

| | |
|---|---|
| **What happened** | A regeneration on a widened configuration passed every sample-level check, IMPROVED the calibration objective, and produced a truth run with 99.98 percent errors. Two diagnoses were published before the third was correct. |
| **Not the parents** | The first blamed tail-dominated parents and a guard was added for it. Measured, those parents have a mean of 1.012, a median of 0.773, a ratio of **1.31** against that guard's threshold of 25, and a 1 - 1e-6 quantile at 33. The guard does not fire on the configuration it was written for; `genconfig.max_parent_mean_over_median`'s docstring is corrected in place and says so. |
| **The cause** | `plca.ParentSampler` tabulated the parent CDF on a LINEARLY spaced grid between the truncation bounds. With the upper bound near 1e8 the spacing was about 18,000, so the entire body fell between the first two grid points and the truth run drew from a step function. Fixed with log-spaced tail points and exact quantile endpoints. |
| **The gate that now exists** | `audits/parent_sampler_fidelity.py` compares the sampler's inverse CDF against the parent's own bisection at thirteen probabilities from 1e-6 to 1 - 1e-6, four sizes, both schemes. Shipped and all bounded candidates: worst error **5e-4**. The rejected configuration: wrong by more than 1 percent on **55 percent** of parents, median 37 percent, worst 99.7, all at the 1e-6 quantile where its lower bound sits eight orders of magnitude below the body. |
| **Fix** | Nothing in the manuscript changes: the configuration was reverted and no paper number survives from it. What the METHODS section may now say is that the truth run's sampler is verified against the exact parent quantiles rather than assumed faithful, which is a strengthening a reviewer would value. Decisions 191, 192. |
| **Status** | Resolved. The shipped configuration never triggered the bug, verified rather than assumed. |

## 170. The pedigree matrix produces a NARROWER spread than real ECC data, not a wider one

| | |
|---|---|
| **The sourcing gap, now closed** | Decision 124 asked for a sweep over the range the pedigree matrix produces. Its factor table was not in `refs/`, so Stage 2h swept the spread relative to each category's own and recorded the sourcing as owed. The author supplied Muller, Lesage, Ciroth, Mutel, Weidema and Samson (2016), Int J Life Cycle Assess 21:1185-1196. |
| **The arithmetic** | Every factor in that table is a contributor to the SQUARE of the geometric standard deviation: `sigma_95 = sqrt(sum of [ln UF_i]^2, plus the basic)` and `GSD = exp(sigma_95 / 2)`. Quoting the combined factor AS a GSD would double the spread. |
| **What it spans, all 3,125 score combinations** | best (1,1,1,1,1) **GSD 1.0247**; median combination 1.2416; worst (5,5,5,5,5) **1.5873**. A median real ECC category is **1.8712**. |
| **The finding** | A pedigree model is systematically NARROWER than the data it stands for, and **61.9 percent of real categories are wider than its worst possible score**. End to end the matrix spans a factor of 1.55; the real arm spans 1.01 to 50.9. |
| **Why that is not a defect in the matrix** | It quantifies uncertainty about ONE datum for ONE process, not the spread of products within a material category, which is what an ECC dataset measures. The paper should say so rather than present the two as rival estimates of one quantity. |
| **What it does to the stage's own sweep** | Taken on the EXCESS over 1, which is how `audits/judgment_arm.py` scales it, the reachable ratio on a median category is 0.028 to 0.674, so of the six ratios swept only **0.5** is attainable. The wide end is a sensitivity and must not be labelled a pedigree model. It STRENGTHENS the stage's conclusion that a judgment model's centre decides everything and its spread barely matters, since the reachable band is narrower than the flat range already measured. |
| **Two reading errors recorded so they are not repeated** | A note from that paper read "GSD 1.279 basic to 1.690 at scores 5,5,5,5,5"; those are the posterior factors for ONE indicator at scores 2 and 3 for the manufacturing sector, neither GSDs nor a range. And a first version of the audit used a straight GSD ratio against a sweep that scales the excess, giving 0.55 to 0.85 and naming the wrong swept points as reachable. |
| **Fix** | State the range, state that the matrix is narrower than the data, and cite the source. `judgment.PEDIGREE_GSD` carries the three computed values. Decision 194. |
| **Status** | Resolved as a measurement. |

## 171. The bandwidth rule measured through the simulation, not only at the fit

| | |
|---|---|
| **The question** | Entry and decision 188 answered the bandwidth sensitivity on the FIT. The author asked for it on the answer: "ultimately, pLCA results are most important, so should we measure those?" |
| **Measured** | 2,000 pLCA groups against the true parents, mean relative error over five outputs, percent. Scott **25.879 / 25.418**; pure Silverman 25.283 / 24.965; the shipped guarded rule **25.473 / 24.819**, for equal weights and Dirichlet shares respectively. |
| **The control** | All four parametric methods are bit-identical across the three bandwidth rules, which says the measurement picks up the bandwidth and nothing else. |
| **The finding** | Scott is worst on every one of the five outputs under both weightings. The guard costs 0.19 of a percentage point under equal weights and BUYS 0.15 under Dirichlet shares, so at the decision level it is free where at the fit level it costs a little. |
| **Fix** | Nothing changes. State that the manuscript's own configuration used Scott, so its numbers understate the kernel estimate on every downstream output -- the conservative direction for this paper's recommendation, and better said than quietly corrected. Decision 195. |
| **Status** | Resolved. |

## 172. THE CORPUS IS REGENERATED. Every synthetic number in the paper moves, and no recommendation does

| | |
|---|---|
| **What changed** | `corpus_2026-09-25` replaces `corpus_2026-09-21`. `min_q1_over_iqr` 0.5 to 0.2 and `cv_log10_mean` 0.129 to 0.329, widening the dispersion the corpus never matched. Generation was reopened by author decision; decisions 47, 48, 55 and 138 are superseded on whether this is possible. |
| **Why it became possible** | The trade that blocked it across three stages -- widening always inflated the uniform-to-variable Wasserstein distance past the real arm's -- was an artifact of the two arms weighting by different rules, and it vanishes once they share one. Entries 160, 168. |
| **The corpus now** | Objective 0.2216 to **0.1862**, which is 5.4 seed standard deviations. Dispersion distance 0.409 to **0.247**; weighting distance 0.340 to **0.151**. Both improved together, which had never happened. |
| **THE NEW WORST CHARACTERISTIC** | `fit_lognorm_SF` at **0.341**, up from 0.233, replacing dispersion. **The manuscript's limitation paragraph currently names dispersion and must be rewritten around this instead.** It is the characteristic measured as moving the method comparison by 0.001 to 0.002, so it is the right thing to have spent -- but it is the number a reviewer points at first. `crit_bw_1` also rises, 0.158 to 0.266, worth 0.006 to 0.015. |
| **Gates** | Parent-level fidelity 5.1e-4 on every quantile of every parent; end-to-end smoke test passed with the mean draw from the true parent at 1.0000; the 1,000-dataset draft predicted the production objective to within 0.002. |
| **THE READING TRAP** | Every ABSOLUTE distance rises -- the six W1 columns by 16 to 45 percent, the error in a material's estimated contribution by 25 to 35. That is scale, not degradation: a more dispersed dataset has a wider parent and a larger absolute distance to it, and divided by each dataset's own spread the distance to the truth is **0.965** times the old corpus's. **Any sentence quoting one of those figures in absolute units must be restated from the new tables, and the paper should say why they rose.** |
| **Fix** | Every synthetic number in the manuscript is recomputed from the new tables. Decision 197. |
| **Status** | Done in the analysis. The manuscript has not been touched. |

## 173. The practitioner rule reproduces on a second, differently built corpus

| | |
|---|---|
| **Why this matters** | The rule -- use a kernel estimate above about 80 declarations, with market-share weights if you have them, a three-parameter lognormal below -- was calibrated on a corpus whose dispersion was short by a factor of nine. Reproducing it on one built with different generator settings is a robustness check the paper could not previously offer. |
| **The threshold** | **81 declarations, unchanged**, with the indistinguishable band 68 to 106 against 68 to 97 before. Penalty for missing it: 6.01 points of extra error at a threshold of 24, 0.10 at 81, 1.27 at 138, 5.96 at 304. |
| **Which method is closest to the truth, by size** | New corpus, with the old figures in brackets. KDE equal weights 34.7 [33.8] / 21.0 [22.5] / 24.9 [28.8] / 21.8 [23.5] across n = 3-9, 10-99, 100-999, 1000+; KDE Dirichlet 22.4 [22.5] / 21.1 [20.0] / 38.0 [42.8] / 67.0 [69.6]; lognormal equal 22.6 [20.9] / 27.2 [25.2] / 12.2 [8.8] / 1.5 [0.6]; lognormal Dirichlet 10.6 [9.9] / 20.5 [17.9] / 22.2 [16.8] / 9.6 [6.1]. **Every ordering identical; nothing moves more than five points.** |
| **The truth run SHARPENS** | The four non-normal methods span **0.0811 to 0.0841** on a material's chance of leading, a 3.7 percent spread against 6.8 before -- MORE indistinguishable than the paper says -- and the normal's penalty over the best grows from 44.2 to **50.6 percent**. |
| **Fix** | **NOTHING. This is INTERNAL VERIFICATION and does not belong in the manuscript.** Author decision, 2026-09-25: "I thought we'd just focus on this one because it's more right. I don't think generating two corpora of data should be a major part of our methodology." Correct on both counts -- the superseded corpus is the less representative one, and describing two would invite a reviewer to ask why the worse one is shown at all. The paper describes ONE corpus, the current one. This entry exists so that a later session knows the rule was checked against a differently built corpus and does not re-run it. |
| **Status** | Measured, and deliberately NOT a manuscript item. |

## 174. The upper truncation is NOT adopted and is stated as an available remedy

| | |
|---|---|
| **The author's decision, 2026-09-25** | "I don't think we need to adopt this. We just need to note in the manuscript that truncation is an option that's very easy to apply if you're dealing with extreme values." |
| **Why that is the right call** | Adopting it would move every number in the study a second time, for a failure mode two existing safeguards already keep out of the results: the profile-likelihood guard bounds the lognormal at fitting time, and the tail term in the criterion charges for whatever survives. Across 60,000 fits the mean charge for mass beyond the scoring grid is 0.0000 to 0.0001. |
| **What was measured, so the sentence can be specific** | Capping every fitted model at two to three times the largest observation cuts the worst runaway fit from 5.4 times the data's own spread to about 1.8, and against the true parent it is not a cost -- it is better in the fourth decimal. It does not pull the two arms apart (0.12 of a percentage point at a cap of 2x) and it does not flatter one family once scored against the truth: the in-sample gain falls entirely on the lognormal, and against the truth the two lognormals move by +0.57 and -0.85 percent while the kernel estimate and the normal move by nothing. |
| **Fix** | One or two sentences, in the discussion rather than the methods, saying that a goodness-of-fit distance between cumulative curves cannot see how far out a model puts its rare values; that this study charges for it with a tail term and watches it with the fitted-model spread ratio; and that a practitioner facing extreme values can truncate each fitted model at a plausible multiple of the largest observation, which removes the failure mode outright at the cost of one assumption and, measured here, costs nothing in accuracy. |
| **Status** | Decided. Not adopted in code; `families.TruncatedAbove` and `cap_models` stay available and tested. Decisions 152, 182, 196. |

## 175. The weighting vocabulary is settled: "market weights" against "uniform weights"

| | |
|---|---|
| **The decision, author, 2026-09-25** | "Market weight vs uniform weight seems right to me. Let's just make sure we're consistent everywhere." |
| **The churn this ends, recorded so it is not repeated** | The stored values are "Uniform" and "Variable". "Variable" was replaced because it reads as "market shares were accounted for", which overclaims -- nobody publishes them (decision 160). Its replacement, "Dirichlet shares", was accurate and inaccessible. Its replacement, "sampled market shares", was accurate and awkward on an axis. **"Market weights" is what the paper uses, everywhere.** |
| **AND THE OVERCLAIM DECISION 160 GUARDED AGAINST IS REAL, so it moves into the text** | "Market weights" alone could be read as real production volumes, which this study does not have. Two things prevent that and neither is a label: the methods section says AT FIRST USE that market weights are drawn from a Dirichlet because production volumes are not published, and the oracle scheme is labelled "known market shares" so the contrast between a drawn weight and a known one is visible wherever both appear. |
| **Fix** | Use "market weights" and "uniform weights" throughout, and "known market shares" for the oracle. Add the first-use sentence in the methods. The stored `method` values do NOT change -- they are the join key for every table and fixture, and renaming them would move numbers for a presentation fix. |
| **Status** | Implemented in `fitting.WT_DISPLAY`, pinned by a test. Stage 3 carries the labels into the remaining figures. |

## 176. The corpus under-represents multimodal-AND-dispersed categories, and that is stated rather than fixed

| | |
|---|---|
| **The limitation, in the form the paper should state it** | The synthetic corpus spans the modality of real material categories and the dispersion of real material categories, and under-represents their INTERSECTION. Multimodal-and-dispersed is **1.3 percent of the corpus against 5.4 percent of the 130 real categories**; conditional on being dispersed, a real category is multimodal **21.2 percent** of the time against the corpus's **14.7**. What the study cannot speak to is roughly one real category in twenty. |
| **Measured on the weight-invariant columns** | The unweighted coefficient of variation and the visible mode count at the fitted bandwidth, with "dispersed" meaning above the real arm's own upper quartile. Weighted versions of these move with the weight rule and are not comparable across the Stage 2h change. |
| **The lever was found and rejected on a measured trade** | `genconfig.mode_share_alpha` at 1 rather than 10 takes the corpus's multimodal share from 0.200 to **0.242 against a real 0.246** -- essentially exact -- and the conditional from 0.147 to 0.161. It costs the arm-to-arm distance on `w_v_uw_wasserstein` **0.172 to 0.275**, a 60 percent degradation on the characteristic the paper is built on. Author decision: keep 10. |
| **The mechanism, which makes this a limitation rather than an unknown** | The conditional is HIGHEST on the narrow superseded configuration and falls in every widening candidate, because widening the components blends the humps together. Three candidates combining `alpha = 1` with more separation made it worse still (0.139, 0.111, 0.136). Separation destroys the conditional; hump-share concentration does not. The conditional and the dispersion marginal are in direct tension in this generator. |
| **What the paper should NOT say** | That the shortfall is unfixable. It is fixable and the price is known and was declined. Say what it costs. |
| **Fix** | One paragraph in the limitations, with the four numbers above. Decision 203. |
| **Status** | Decided. Not fixed, deliberately. |

## 177. A NEW RESULT THE MANUSCRIPT DOES NOT CONTAIN: choosing the method per material

| | |
|---|---|
| **Manuscript** | Compares six fixed UQ methods and, from Stage 2f onward, recommends a size rule for WHICH ONE to use. It nowhere considers using a different method for different materials inside one probabilistic LCA, because every pLCA the study ran fitted one method to all four. |
| **Code** | Stage 2j measures it. `src/mixedpolicy.py` and eight cells at the end of notebook 3. The rule is the study's own and has one number in it: a kernel estimate with market weights at or above 81 declarations, a three-parameter lognormal with uniform weights below. |
| **What it found** | The rule is the CLOSEST of the seven policies on ALL SIXTEEN claims a probabilistic LCA makes, by a median 11.3 percent of the best fixed policy's own error, range 3.4 to 16.8 percent, every paired interval clearing zero. On the fit it is worth +14.2 percent [12.4, 16.1]; on the claims a median +11.3. |
| **Why it matters for the text** | **It resolves the paper's widest gap between a fit result and a decision result.** The manuscript will say that a kernel estimate overtakes a three-parameter lognormal on goodness of fit at about 81 declarations and does not pull clear on the downstream claims until roughly ten times that. The mechanism is that a pLCA picks one method for all four materials, so one material's advantage is averaged against three neighbours drawn at random. **Under a per-material rule there is nothing to average against, and the attenuation very largely disappears.** |
| **Fix** | **Text, new.** A results subsection and a sentence in the recommendation. Four qualifications must travel with it and are in `reports/STAGE_REPORT_2j.md`: above the threshold the rule is wrong on the LEVEL claims where every material is large; on the argmax it is third of seven while on the continuous version of the same question it is first; it is the most accurate policy per decision and not the least biased at building scale; and it captures about a fifth of what an unreachable per-material oracle could buy. |
| **Status** | OPEN. **Whether the paper recommends it is an author decision**, and so is whether the scorecard figure gains a seventh column -- which is not free, because `best_method` and `stakes` there are properties of the SET of policies compared and would change on all sixteen rows. |

## 178. The six-method scorecard is not superseded by the seven-policy one

| | |
|---|---|
| **Manuscript** | Will print the sixteen-claim scorecard for six methods, with a black box on the closest method in each row and a bar for what the choice costs. |
| **Code** | Stage 2j writes a SECOND scorecard, `TABLE_MixedPolicyScorecard.csv`, over seven policies. The size rule is the best on all sixteen rows of it. |
| **Fix** | **Text.** Do not merge them without saying so. The six-method table answers "which METHOD is best", and `best_method`, `stakes` and `excess` in it are properties of the six-policy set; the seven-policy table answers "is a per-material POLICY better than any fixed method", and its own `stakes` column is a different quantity. Quoting a number from one as though it came from the other is the kind of mixing decisions 157 and 174 already had to correct twice. |
| **Status** | OPEN, and it is a presentation decision rather than an analysis one. |

## 179. The cutoff is a RANGE, and the gain is the weighting switch

| | |
|---|---|
| **Manuscript** | Will recommend a kernel estimate above about 81 declarations and a three-parameter lognormal below, as a single number read off the goodness-of-fit curve. |
| **Code** | Stage 2j sweeps thirteen cutoffs from 20 to 220 at the CLAIM level and scores four one-axis variants of the rule. |
| **What changes** | **Print a range, not 81.** Best cutoff 70; 50 to 81 statistically indistinguishable; everything from 40 to 130 within 0.0011 of the best on a level of 0.20; the fit-level optimum at 81 sits inside it. Across the whole sweep the cutoff is worth 0.47 points against 2.68 for the rule itself. **And describe the rule by what does the work**: switching only the WEIGHTING at the cutoff recovers three quarters of the gain, switching only the FAMILY recovers nothing. |
| **The weighting sentence the paper needs** | Market weights are better than uniform only above about 81 declarations -- the market-weighted fit is closer to the truth on 32.8 percent of datasets at 3 to 9 declarations, 53.6 percent at 81 to 99 and 78.7 percent above 1,000 -- because the study's market weights are a flat-Dirichlet GUESS. Knowing the shares beats ignoring them everywhere (oracle 0.1267 against uniform 0.1670 for the lognormal). The rule's value is knowing when guessing is worth it. |
| **Status** | OPEN. Text, new. The width of the printed range is the author's wording; the measurement supports roughly 50 to 100. |
