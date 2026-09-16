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
| **Code** | `random_logcount(lo=4, hi=1000)`, so the lower bound is 4, not 3. More importantly, the metric outlier filter removes 824 datasets on `n`, capping the effective maximum at **749**. No dataset larger than 749 survives into the analysed set of 10,000. |
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
| **Fix** | **Undecided, and it needs a decision.** Either the divisor is a deliberate conditional normalization, in which case the manuscript must define the quantity as a percentage of the iterations in which the cap actually binds, or it is a patch, in which case the code should change. The matching material-reduction block has no such divisor, so the two reduction strategies are normalized differently while the manuscript states their definitions are "equivalent". |
| **Status** | Open. Deferred to Stage 2 as a statistical judgment item. |

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
| **Consequence** | It removed exactly the cases the paper exists to study. `weight_outliers` flagged more datasets than any other metric (1,061), and the removed set averaged 0.0755 against 0.0146 for the kept set, a difference of 0.83 pooled standard deviations. The removed datasets were also less normal (fit_norm_SW -0.89 sd), more variable (coeffvar +0.48 sd), larger (n +0.52 sd), more multimodal (+0.46 sd) and had a larger uniform-to-variable W1 (+0.54 sd). The filter flagged 824 datasets on `n` alone, every one with n >= 750, which is what capped the analysed maximum at 749 against a stated 1,000. It also flagged 14 datasets on `mean_uw`, a column that is 1.0 by construction and ranges only from 0.99999999999999911 to 1.0000000000000009. |
| **Fix** | **Code, done in Stage 2a.** Replaced by a validity-only filter that rejects a dataset solely because it cannot be analysed. Empirical plausibility is now a reported coverage statistic, not an enforced criterion. |
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

## CURRENT CANONICAL NUMBERS, as of Stage 2b closing, 2026-09-14

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
| coefficient of variation | 0.658 | 0.006 | **6.929** (`Aggregates`; was 13.404 for `PowerCabling` before decision 63 removed two mislabelled records) |
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

`corpus_2026-09-14d`, seed 42. **9,999 datasets, not 10,000** (one parent failed
to solve and was reported rather than approximated), plus a 50-dataset probe set
held outside every aggregate. Stratified 2,500 per stratum over n = 3-9, 10-99,
100-999 and 1000-9999.

Match against the 149-dataset arm: mean standardized W1 across the ten
characteristics **0.2326**; visible-mode total variation **0.0128**. Worst
characteristic `fit_lognorm_SW` at 0.386, then `entropy` 0.321, `fit_norm_SW`
0.286, `coeffvar` 0.274.

### What Stage 2b changed, and the numbers it adds

| | |
|---|---|
| **empirical values** | 117,090 to **117,079**, and six datasets move. Entry 35 |
| **pLCAs** | **2,499, not 2,500**, over groups of four covering 9,996 of 9,999 datasets. Entry 39 |
| **W1, empirical, mean rank over the six methods** | `Lognormal, Variable` **2.09**, `KDE, Variable` 2.74, `Lognormal, Uniform` 3.34, `KDE, Uniform` 3.52, `Normal, Variable` 4.21, `Normal, Uniform` 5.11 |
| **W1, synthetic, mean rank** | `KDE, Variable` **2.12**, `Lognormal, Variable` 2.19, `Normal, Variable` 3.64, `KDE, Uniform` 3.93, `Lognormal, Uniform` 4.17, `Normal, Uniform` 4.95 |
| **W1, empirical, mean** | `Lognormal, Variable` 0.178, `Lognormal, Uniform` 0.211, `KDE, Variable` 0.251, `KDE, Uniform` 0.286, `Normal, Variable` 0.436, `Normal, Uniform` 0.490 |
| **W1, synthetic, mean** | `Lognormal, Variable` 0.0988, `KDE, Variable` 0.1017, `Lognormal, Uniform` 0.1676, `KDE, Uniform` 0.1711, `Normal, Variable` 0.1724, `Normal, Uniform` 0.2187 |
| **the lognormal** | 3-parameter, threshold by profile likelihood, guard at 0.25 weighted standard deviations below min(x); the +0.5 offset is retired. Entries 37, 40, 43 |
| **the support** | (0, inf) open at zero, every method truncated and renormalized, sampling by inverse CDF. Decision 13, confirmed |

**The empirical W1 values above are NOT comparable to anything in the manuscript**,
which reports the same quantity in raw category units. Entry 41.

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

**THE DECISION, stated as three options.** Earlier versions of this entry said
"undecided and it needs one" without saying what was on offer, which is a defect
in the document rather than a hard question.

| option | what it means | cost |
|---|---|---|
| **A. Change the text** (recommended) | State coverage as measured and name the exceptions. The claim becomes: the corpus covers the empirical characteristic space with margin except at the extreme upper tail of dispersion, where 5 of 149 datasets sit beyond it, and above 9,999 values per dataset, which the probe set covers by design | nothing; no regeneration |
| B. Widen the generator and regenerate | **MEASURED AND NOT AVAILABLE AS A PARAMETER CHANGE.** Eight candidates were swept in `audits/dispersion_reach.py`: raising the target centre by 0.4, the spread by 1.8x, the upper truncation to 60, and relaxing the quartile-ratio floor `min_q1_over_iqr` from 0.5 through 0.1, 0.05 to 0.01. **The achieved sample coefficient of variation moves from 1.65 to at most 2.15**, against an empirical maximum of 14.34, and NONE of the eight puts a single dataset above 3 | reaching the empirical tail needs heavier-tailed parents or a different truncation rule, which is a generator REDESIGN, not a retune and not one regeneration |
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
| **The general point, which outlives this particular fix** | **Every method in this study is selected by a CDF distance and then used as a sampler, and no part of the analysis currently checks that the selected model is a sane sampler.** The lognormal is where it bit because that family has an unbounded likelihood; nothing rules out the same failure elsewhere, and the KDE's own tail behaviour under a bandwidth sweep is a Stage 2h question that has the same shape. |
| **Fix, the general version** | **Analysis, and it needs an owner.** Stage 2c owns the evaluation target and should decide whether W1 alone is a sufficient criterion or whether a tail-sensitive companion is needed; Stage 2g owns the downstream metrics and meets the same question from the other end. A cheap first step is to report, for every fitted model, the ratio of its own standard deviation to the data's. |
| **Status** | Fixed for the lognormal. **Open as a general question, owner unassigned, and it is the kind of thing a reviewer finds.** |

## 44. WHY the KDE loses on the empirical arm, and why two of the three reasons are defects in the comparison rather than facts about KDE

| | |
|---|---|
| **Context** | Entry 40 records that the KDE is not the best-fitting method on the 149 empirical datasets. That is a statement about numbers. This entry is the diagnosis, and it matters far more, because **two of the three mechanisms are defects in how the comparison is SPECIFIED.** `audits/why_kde_loses.py`. |
| **1. DATASET SIZE, and the two arms do not actually disagree** | The KDE's mean rank improves monotonically with n on BOTH arms and the lognormal's degrades. Empirical, `KDE, Variable` by size band: 3.60 at n = 3-9, 2.77 at 10-99, 2.54 at 100-999, **1.64 at n >= 1000**, where it is second of six and `KDE, Uniform` is first at 1.36. `Lognormal, Variable` runs the other way: 2.55, 1.76, 2.08, 3.64. The synthetic corpus shows the same pattern. **The corpus allocates 2,500 datasets to each size stratum, so 25 percent sit above n = 1,000, against 7.4 percent of the empirical arm** -- the empirical mix is 13.4 / 53.0 / 26.2 / 7.4 percent, median n = 50. Reweighting the corpus to the empirical size mix moves `KDE, Variable` from 2.12 to 2.25 and `Lognormal, Variable` from 2.19 to 2.02, **which is the empirical ordering**. The arms agree; equal allocation is a precision choice, not a claim about how common each size is. |
| **2. THE BANDWIDTH RULE, and this study does not use the author's own** | `BW_METHOD = 'scott'`, `1.06 * sd * n_eff ** -0.2`, oversmooths right-skewed data: median bandwidth / sd of **0.56** on the empirical arm under variable weights. A Gaussian KDE inflates the fitted variance by `sqrt(1 + (h/sd)^2)`, so that is 15 percent of spread the data does not have, and it is why the fitted KDE puts a mean of 9.4 percent of its mass below zero on an arm whose support is (0, inf). Silverman's robust rule, `0.9 * min(sd, IQR/1.34) * n_eff ** -0.2`, gives 0.33 and 5.2 percent. **Silverman beats Scott on 100 percent of the 149 datasets under variable weighting** (95.3 percent under uniform): mean W1 0.2507 to **0.1228**, against the lognormal's 0.1778. Under Silverman the KDE beats the lognormal on **81.2 percent** of datasets; under Scott, on 33.6 percent. **Torres et al. (2026), the author's KL2 paper, uses Silverman and justifies it explicitly. This paper uses Scott.** Entry 10 and decision 9 already flag the inconsistency; this is what it costs. |
| **3. THE CRITERION REWARDS UNDERSMOOTHING, which confounds mechanism 2** | W1 falls monotonically as the KDE bandwidth shrinks: mean W1 over the empirical arm at multiples of Scott's bandwidth is 0.2507 at 1.0, 0.1365 at 0.5, 0.0497 at 0.1, **0.0415 at 0.05**, 0.0444 at 0.02. The minimizing multiple is 0.02 for 95 of 149 datasets and 0.01 for 26 more; it turns up only at 0.01, and that is the 1,000-point scoring grid running out of resolution rather than a real optimum. A KDE with a vanishing bandwidth IS the empirical distribution it is being scored against. **So W1 cannot arbitrate between methods of different flexibility, and the Silverman result above is partly just "Silverman is smaller".** |
| **What survives the confound** | The direction, and it is the important part. Scott oversmooths this data badly, and **the KDE still loses to the lognormal under Scott even though the criterion is biased in the KDE's favour.** Both of those are real. What is NOT established is the magnitude of the KDE's disadvantage, or whether it has one at all under a defensible bandwidth and a criterion with a complexity penalty. |
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
| **The fix** | `weighted_bw(..., bw_method='silverman_guarded')`: Silverman above an effective-sample-size threshold, Scott below it. `SILVERMAN_MIN_NEFF = 30`. It uses the KISH EFFECTIVE sample size, not raw n, because a concentrated Dirichlet draw can leave three effective observations in a 200-value dataset. |
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
