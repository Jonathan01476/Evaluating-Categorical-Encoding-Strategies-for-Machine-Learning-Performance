# Comparing airport encodings for arrival delay

How much does the choice of airport encoding change prediction error for the same model? Predict arrival delay in minutes immediately after departure, when departure delay is known. For each model, compare five ways to represent origin and destination while holding the flights, other predictors, and model settings fixed.

## 1. Prepare the same flights and predictors

**Completed in the notebook.** The dataset builder now returns only the selected encoding of origin and destination plus the target. The modeling predictors are limited to location. Model fitting and evaluation remain to be done.

Use all 327,346 flights with an observed `arr_delay`. Keep early arrivals and extreme delays. Set aside the target and build the predictor matrix from the following features only.

| Shared predictor | Preparation |
| --- | --- |
| Departure delay and distance | Use the recorded numeric values. |
| Carrier, month, and weekday | Derive weekday from the flight date. Fit one shared one-hot encoder on training rows and reuse its columns across all five datasets. An unseen category gets zeros for that field. |
| Scheduled departure and arrival times | Validate each HHMM value and convert it to minutes since midnight, treating 2400 as midnight. Represent each time with sine and cosine over a 1,440-minute cycle. |

Add one airport representation to these shared features. Keep the flight ID for matching predictions. Actual arrival time and air time are unavailable at departure and must stay outside the predictor matrix.

## 2. Build each encoding within the training fold

Reproduce the notebook's random 80/20 split with seed 42. Reserve the 20% for final testing. On the other 80%, create five shuffled cross-validation folds with seed 42. Save the row assignments once and use them for every run.

For each fold, pass its **raw training and validation rows** to `make_encoded_datasets`. Fit learned transformations on the training rows. Building fresh encodings for each fold prevents validation outcomes from influencing airport features.

| Airport encoding | Representation for origin and destination | Airport absent from training |
| --- | --- | --- |
| One-hot | One indicator per observed airport in each field | All zeros for that field |
| Ordinal | Integer codes in alphabetical order | -1 |
| Frequency | Share of training flights with that airport in the field | 0 |
| Smoothed target | Mean arrival delay by airport, shrunk toward the training mean with weight 20 | Training mean delay |
| Geographic | Latitude and longitude from the fixed lookup | Use the lookup and flag any missing coordinates as an error |

For smoothed target encoding, use `fit_transform` with five internal folds and seed 42 so each training row's value excludes outcomes from its internal fold. Use the encoder's `transform` for validation and final test flights.

## 3. Fit the models

Use the same settings for all five encodings within each row of the table. Store predictors as float64 CSR matrices. For Ridge, fit `StandardScaler(with_mean=False)` on training rows. Use unscaled features for the tree models.

| Model | Fixed settings |
| --- | --- |
| Ridge regression | Alpha 1.0, intercept enabled, LSQR solver, tolerance 0.0001, maximum 10,000 iterations |
| Random forest | 200 trees, maximum depth 20, minimum 5 samples per leaf, all features available at each split, bootstrap sampling, seed 42, one worker |
| Gradient boosting | Squared error, 200 stages, learning rate 0.05, maximum depth 3, minimum 5 samples per leaf, all features and rows per stage, early stopping disabled, seed 42 |

Include four baselines. One predicts the training mean delay. The other three fit each model using only the shared predictors, which shows how much airport information adds.

First run all 15 model and encoding combinations on a small development sample. Check row and target alignment, finite predictions, unknown-category handling, and sparse matrix compatibility. Change validation targets and confirm that training features and validation predictions stay identical. Then run the five full folds with the fixed settings.

## 4. Compare predictions on the same flights

Use **root mean squared error (RMSE), in minutes**, as the primary score. Also report mean absolute error (MAE) in minutes and R². Within each model, subtract one-hot's error from each encoding's error on the same flights. A difference of −0.5 minutes means the encoding reduced that error metric by half a minute.

- Report each fold's scores, their mean and standard deviation, and the scores calculated over all out-of-fold predictions together.
- Compare each model with its no-airport baseline and the mean-delay baseline. Examine errors on long delays and at airports with few training flights. Include subgroup counts.
- Record feature count, matrix memory, and time spent preparing features, fitting, and predicting. Use the same machine and thread limits and count shared preprocessing costs consistently.
- If reporting uncertainty in paired error differences, resample whole flight dates together across encodings (2,000 draws, seed 42). These intervals describe variation in the held-out predictions conditional on the fitted models.

## 5. Evaluate the holdout and save results

Select the preferred combination by development RMSE, breaking ties by MAE and then fit time. Lock that choice before scoring the test set. Refit the 15 combinations and four baselines on all development rows, evaluate them on the same test flights once, and report every result.

Save fold scores in `results/cv_metrics.csv`, final scores in `results/test_metrics.csv`, and paired differences in `results/paired_comparisons.csv`. Keep row-level predictions with flight ID, date, fold, model, encoding, observed delay, and predicted delay. Record split assignments, input hashes, feature order, seeds, model parameters, package versions, hardware, thread limits, and timings so the notebook can reproduce the comparison from a fresh kernel.

Interpret the findings within each model family and for flights with recorded delays in this historical sample. The random split measures performance on similar flights. Geographic encoding supplies external coordinates, and each representation can change how many features the model sees.
