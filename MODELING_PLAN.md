# Controlled comparison of airport encodings

## Objective and scope

Predict `arr_delay` in minutes and measure how representing `origin` and `dest` changes regression performance. Negative outcomes mean early arrivals. This follows the notebook's selected target; predicting an arrival timestamp would be a separate task requiring date and time-zone handling.

The primary question is: **within a fixed model family, how does changing only airport encoding affect held-out error?** Compare one-hot, ordinal, frequency, smoothed target, and geographic encoding. Model-family comparisons are secondary; report encoding-by-model interactions rather than claiming one universally best encoding.

This plan extends the current local `Final_Project.ipynb`, including `make_encoded_datasets`, `raw_train`, and `raw_test`. The existing encoded datasets are useful demonstrations and can support the final train/test fit. They must not be reused as precomputed inputs to cross-validation. This document plans the modeling work; it does not report completed model experiments.

## 1. Freeze the prediction task and common inputs

- Prediction moment: immediately after departure, when `dep_delay` is known. This is an explicit experimental assumption, not a pre-departure forecasting claim.
- Retain all rows with observed `arr_delay`, including negative and extreme delays. The notebook documents 327,346 eligible flights from 336,776 source rows; verify these counts at execution.
- Use one shared predictor allowlist: `dep_delay`, `distance`, `carrier`, month, day of week derived from `year/month/day`, and sine/cosine pairs for scheduled departure and scheduled arrival time. Add exactly one representation of `origin` and `dest`.
- Convert scheduled HHMM to minutes since midnight, mapping 2400 to 0 for these cyclic features, then use sine/cosine with a 1,440-minute period. These are local clock features, not elapsed flight duration. Validate values before conversion.
- One-hot encode `carrier`, month, and day of week using training-only vocabularies and the same unknown-category rule in every experiment. Do not give ordinal ordering to month or weekday.
- Exclude `arr_delay` from X. Exclude actual `arr_time` and `air_time` because they are unavailable at the prediction moment. Exclude `id`, `tailnum`, flight number, `name`, actual `dep_time`, raw date/time fields, `hour`, and `minute`; identifiers, duplicate information, and alternative airport proxies are outside this experiment. Retain original row IDs only for alignment and split records.
- Keep the outcome in minutes with no transformation, trimming, balancing, or encoding-specific filtering. Results apply to flights with recorded arrival delay, not cancellation prediction.

## 2. Freeze the splits before fitting

1. Reproduce the notebook's shared 80/20 random holdout with `random_state=42` on eligible raw rows, preserving source row order. Save train/test row IDs and data-file hashes so the split can be checked on rerun.
2. Keep the 20% test partition out of feature decisions, model selection, and tuning. Existing exploratory summaries are not an untouched external validation set; disclose this limitation.
3. On the 80% development partition, generate one `KFold(n_splits=5, shuffle=True, random_state=42)` split list and reuse its exact row assignments for all models and encodings.
4. Assert train/validation/test disjointness, target/index alignment, and identical eligible rows across all runs. Do not independently split each encoded dataset.

The random holdout estimates performance on held-out flights from the same historical population. It does not establish future-flight performance. A chronological assessment is a separate sensitivity experiment with its own shared splits, and must never replace only one encoding's evaluation.

## 3. Fit preprocessing inside every fold

For each development fold, take raw training and validation rows, then call `make_encoded_datasets(fold_train, fold_validation, airport_coordinates, target_column="arr_delay")`. Fit all remaining preprocessing only on fold-training rows. Reuse the resulting five representations across model families within that fold.

Keep the notebook's airport rules fixed:

| Encoding | Fixed rule | Unseen airport |
| --- | --- | --- |
| One-hot | Training vocabulary, no dropped reference category | All zeros |
| Ordinal | Alphabetical training vocabulary | -1 |
| Frequency | Training category count / training row count | 0 |
| Smoothed target | Continuous target, smoothing 20, internal five-fold cross-fitting, seed 42 | Training target mean |
| Geographic | Fixed local latitude/longitude lookup, four features | Fail validation if coordinates are missing |

For target encoding, training rows must use `fit_transform`, which excludes each row's internal fold when constructing its encoding. Validation/test rows use `transform` from the training-only fitted encoder. Never encode the full development partition before outer cross-validation. Internal target-encoding folds are distinct from the five model-evaluation folds.

Select X through the allowlist plus the returned airport feature names, rather than simply dropping the target from the encoded frames: those frames preserve all other original columns. Hold y separately. Fail clearly on unexpected missing/nonfinite predictors; the current eligible subset is documented as complete. Any later imputation policy must be specified once, fitted within folds, and applied to every encoding.

Use one matrix policy for all variants: float64 CSR matrices, with feature-name order recorded. Scale every predictor with training-fitted `StandardScaler(with_mean=False)` for Ridge; use unscaled features for both tree models. This preprocessing difference is fixed by model family, never chosen by encoding. Do not enable native categorical handling or add interactions for just one representation. Check estimator compatibility and memory use in a common smoke run before the full experiment.

## 4. Primary experiment: fixed model configurations

Use these prespecified configurations across all five encodings. Specify all remaining effective estimator parameters in the run manifest and pin package versions before the full run.

| Family | Configuration |
| --- | --- |
| Regularized linear regression | `Ridge(alpha=1.0, solver="lsqr", tol=1e-4, max_iter=10000, fit_intercept=True)` |
| Random forest | `RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_leaf=5, max_features=1.0, bootstrap=True, random_state=42, n_jobs=1)` |
| Gradient boosting | `GradientBoostingRegressor(loss="squared_error", n_estimators=200, learning_rate=0.05, max_depth=3, min_samples_leaf=5, max_features=None, subsample=1.0, n_iter_no_change=None, random_state=42)` |

Ridge is the regularized linear baseline; label it explicitly rather than calling it ordinary least squares. Tree models consider all columns at a split to avoid a feature-subsampling fraction changing the number considered with encoding dimension. Different dimensions still change model capacity and search behavior; that is part of the representation's effect under the fixed protocol.

Fit `DummyRegressor(strategy="mean")` once per fold as a common reference, with no airport predictors. Also fit each of the three model families on the common non-airport features alone to measure the incremental value of airport information.

The primary encoded experiment has 5 encodings x 3 model families x 5 folds = 75 fits, plus 5 dummy and 15 no-airport fits. Use all eligible development rows. If resource limits require changing the protocol, make one documented change before full scoring and rerun every affected combination. Do not selectively reduce data, tree counts, or folds for slower encodings.

No encoding-specific hyperparameter tuning or early stopping is permitted in the primary comparison. An optional later tuning study must use identical candidate grids, folds, search budgets, and scoring across encodings, and be reported separately as optimized-pipeline performance. Use nested CV if reporting unbiased CV estimates after tuning; do not use the final test set to tune.

## 5. Score paired predictions and record costs

- Primary metric: RMSE in minutes. Secondary metrics: MAE in minutes and R-squared. Lower RMSE/MAE is better; R-squared may be negative.
- Save out-of-fold predictions with row ID, date, fold, encoding, model, observed outcome, and prediction. Every combination must predict exactly the same held-out rows.
- Report each fold's metrics, mean and standard deviation across the five folds, and pooled out-of-fold metrics computed directly from predictions. Do not confuse mean fold RMSE with pooled RMSE.
- Within each model family, use one-hot as a prespecified reference and report paired RMSE/MAE differences on identical folds. Negative differences mean improvement over one-hot. Also compare with dummy and no-airport baselines.
- Treat fold standard deviations as descriptive, not confidence intervals or five independent experiments. If uncertainty intervals are needed, use a paired bootstrap of held-out predictions grouped by flight date (2,000 resamples, seed 42), recomputing metric differences using the same resampled dates for both encodings. Label these conditional intervals; they do not include uncertainty from refitting or guarantee temporal generalization.
- Record preprocessing time, estimator fit time, prediction time, final feature count, and matrix memory separately. Run sequentially on the same machine with fixed thread limits and package versions. If encoded matrices are cached across models, attribute their construction cost consistently rather than giving later models a zero cost. Timing is descriptive unless repeated under the same conditions.
- Diagnose residuals, large-delay errors, and errors by airport support using development out-of-fold predictions. Record subgroup sizes and use identical subgroup definitions. Do not change the primary metric after viewing results.

## 6. Lock decisions and evaluate the final holdout

After development analysis, freeze the protocol and name any selected configuration using development RMSE only (tie-break with MAE, then lower fit time). Rebuild encodings/preprocessing on all raw development rows and fit all 15 fixed combinations, plus the four reference baselines. Target-encoded training rows still require internal cross-fitting; the test set receives development-only mappings.

Evaluate the holdout in one final comparison batch. Publish all combinations, not just the winner. If naming a preferred model, report the development-selected model's test result without selecting a new winner based on test error. No retraining decisions may use the test results; follow-up experiments require a separately labeled protocol and fresh evaluation data where possible.

Interpret differences as effects of encoding **under these models, feature choices, splits, and settings**. Geography adds external spatial information; this is a comparison of representations, not proof that equal information content was supplied to all models. Target encoding includes its fixed cross-fitting and smoothing procedure. Shared settings control important confounders but cannot make effective capacity identical across representations.

## 7. Implementation sequence and deliverables

1. Add a modeling configuration and shared feature-preparation section after the notebook's encoding demonstrations. Verify target, prediction moment, row counts, feature availability, and coordinate coverage.
2. Add a shared split registry and fold-local encoding/preprocessing runner, reusing the existing function. Save row IDs and hashes.
3. Perform a small common smoke run on development data only: check all five encodings and three model families, finite predictions, exact target alignment, sparse-matrix compatibility, unknown categories, and absence of target/post-arrival features. Perturb validation targets and verify training features and validation predictions remain unchanged. Verify that target encoding's internal fit/transform protocol is used.
4. Freeze configuration and run all development fits. Produce paired comparisons, baseline comparisons, runtime/dimension tables, and residual plots.
5. Record development conclusions and lock final choices. Run the final holdout batch and write a limitations-aware discussion.
6. Save `results/cv_metrics.csv`, `results/test_metrics.csv`, `results/paired_comparisons.csv`, prediction files, split assignments, and a run manifest containing seeds, feature names, parameters, dependency versions, input hashes, hardware/thread settings, and timings. Keep large artifacts out of Git unless intentionally selected; commit concise results and reproducible code.

Completion means a clean-kernel notebook run reproduces the shared splits and results; every encoding/model pair is present; no row is silently lost; all learned preprocessing is training-only; and the written conclusion distinguishes fixed-configuration encoding effects from optional tuned performance.

## Plan review and incorporated improvements

The initial outline was: use the existing five datasets, train linear/forest/boosting models with a shared split, compare RMSE/MAE and runtime, then report the best result. Reviewing that outline against the notebook exposed these problems, resolved in the plan above:

| Review finding | Improvement incorporated |
| --- | --- |
| “Arrival time” is ambiguous | Lock `arr_delay` in minutes and state the departure-time prediction assumption. |
| Encoded frames still contain the outcome and post-arrival fields | Use an explicit predictor allowlist and separate X/y. |
| CV on already target-encoded data leaks validation outcomes | Rebuild all encodings inside each raw-data evaluation fold; preserve internal target cross-fitting. |
| Separate splits or dropped rows would confound the encoding comparison | Persist one shared split/fold registry and assert row alignment. |
| Per-encoding tuning changes more than encoding | Use fixed model settings for the primary study and separate any tuning study. |
| Feature counts interact with scaling and tree sampling | Fix scaling by family, matrix policy, and all-feature tree splits; disclose remaining capacity differences. |
| A winner without references may show little useful signal | Add mean and no-airport baselines and paired differences against one-hot. |
| Fold scores are correlated and extreme delays affect RMSE | Report pooled metrics, MAE, descriptive dispersion, paired date-grouped uncertainty if needed, and shared residual diagnostics. |
| Selecting on test scores makes the holdout part of model selection | Lock choices on development data and evaluate the full final comparison once. |
| Random splits could be misrepresented as forecasting | Limit claims to the historical population; reserve temporal evaluation for a separate sensitivity study. |

Review outcome: ready for implementation as a controlled comparison. Remaining limitations are explicit: observational data, one historical population, dependence between flights, excluded missing outcomes, fixed rather than optimal settings, and differing representational capacity. No model-performance claim is made before execution.
