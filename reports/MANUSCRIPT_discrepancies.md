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
| **Fix** | **Text.** Report the sensitivity. See `outputs/tables/stage2a/TABLE_2a_EmpiricalCleaningSensitivity.csv`. |
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
| **Manuscript** | States 138 empirical ECC datasets throughout, and the count appears in figure captions and in the abstract's framing of the empirical arm. |
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
| **Caveat, and it is not small** | Part of the movement is the newer pull rather than the cleaning rule; the two are separated in `reports/HANDOFF_stage-2a2.md` and in `outputs/tables/stage2a2/TABLE_2a2_FourWayComparison.csv`. |
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
| **Status** | Open, awaiting the author's choice of A, B or C. Decision 29 and `CompareUQMethods_FIG_MetricCoverage.png` must be revisited whichever is chosen. |

**THE DECISION, stated as three options.** Earlier versions of this entry said
"undecided and it needs one" without saying what was on offer, which is a defect
in the document rather than a hard question.

| option | what it means | cost |
|---|---|---|
| **A. Change the text** (recommended) | State coverage as measured and name the exceptions. The claim becomes: the corpus covers the empirical characteristic space with margin except at the extreme upper tail of dispersion, where 5 of 149 datasets sit beyond it, and above 9,999 values per dataset, which the probe set covers by design | nothing; no regeneration |
| B. Widen the generator and regenerate | **MEASURED AND NOT AVAILABLE AS A PARAMETER CHANGE.** Eight candidates were swept in `audits/stage2a3/q5_dispersion_reach.py`: raising the target centre by 0.4, the spread by 1.8x, the upper truncation to 60, and relaxing the quartile-ratio floor `min_q1_over_iqr` from 0.5 through 0.1, 0.05 to 0.01. **The achieved sample coefficient of variation moves from 1.65 to at most 2.15**, against an empirical maximum of 14.34, and NONE of the eight puts a single dataset above 3 | reaching the empirical tail needs heavier-tailed parents or a different truncation rule, which is a generator REDESIGN, not a retune and not one regeneration |
| C. Exclude the uncovered categories | Drops `Aggregates`, `Chairs`, `Elevators`, `Grouting`, `PowerCabling` from the arm | reads the ECC values to decide inclusion, and biases the arm toward low dispersion on the exact dimension the study measures. Advised against |

**Why B is not available.** The binding constraint is not the target but the positivity floor of the log truncation rule, `min_q1_over_iqr`, which caps the parent's quartile ratio at `1 + 1/min_q1_over_iqr`: 3 at the current 0.5, against empirical quartile ratios of 4.5 to 284.6 in the five uncovered categories. Relaxing it to 0.01 still reaches a maximum sample coefficient of variation of only 2.15. Eight candidates were swept and none put a single synthetic dataset above 3. Reaching the empirical tail is a generator redesign.

**Why A is recommended.** The five datasets uncovered on dispersion are exactly
the categories the study already identifies as not one product and leaves whole
for that reason. The exception therefore falls where the paper has already told
the reader to expect trouble, and it can be written as one sentence that
strengthens the account rather than weakening it. The three uncovered on `n` are
decision 19 working as designed: the corpus stops at 9,999 values and the probe
set covers above it.

