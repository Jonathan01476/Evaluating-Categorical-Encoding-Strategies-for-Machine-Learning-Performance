# Evaluating Categorical Encoding Strategies for Machine Learning Performance

# Location Encoding and Model Comparison

## Modeling plan

Run [the modeling notebook](src/modeling/Modeling.ipynb) for all 15 model and encoding combinations. See the [modeling plan](MODELING_PLAN.md) for the shared protocol.

## Project Overview

This project investigates how different encoding strategies for categorical location variables affect the predictive performance of machine learning models. Location variables can contain hundreds or thousands of unique categories, making their representation an important modeling decision.

The notebook draws one reproducible sample of 100,000 flights with observed arrival delays (seed 42). All five encoded datasets use the same 80,000 training and 20,000 test flights. Each dataset contains only its encoded origin and destination features plus the arrival-delay target. No other flight columns are included as model inputs.

- **One-hot encoding:** Creates a separate binary feature for each location.
- **Ordinal encoding:** Assigns a numerical identifier to each location.
- **Frequency encoding:** Uses the airport's share of training flights as an origin or destination.
- **Target encoding:** Represents locations using statistics derived from the target variable.
- **Geographic encoding:** Represents locations using spatial information, such as coordinates or distances.

## Methodology

The modeling notebook evaluates Ridge, Gradient Boosting, and a standalone Decision Tree across all five encodings. Each model family uses one fixed configuration. All combinations use the same five development folds, with learned preprocessing fitted inside each fold, and the same held-out test flights.

## Research Questions

1. Which location encoding strategy produces the best predictive performance?
2. Does the optimal encoding strategy depend on the machine learning algorithm?
3. How do different encoding strategies affect computational efficiency and model complexity?

## Evaluation

Compare prediction error using RMSE and MAE in minutes. The notebook also reports R? and wall-clock fit time.

Our goal is to determine whether certain encoding strategies consistently outperform others or whether their effectiveness depends on the model being used.

## Expected Outcome

The project will provide an empirical comparison of location encoding methods and demonstrate how encoding choices interact with machine learning algorithms. Rather than assuming one encoding strategy is universally superior, we will identify which combinations work best for the selected prediction task.

## Data files

All datasets are stored in `data/`:

- `data/flights.csv`: flight records, including the `arr_delay` target.
- `data/airport_coordinates.csv`: airport latitude/longitude lookup for geographic encoding.

Use [the data-preparation notebook](src/Final_Project.ipynb) to inspect the sample and encoded inputs. Both notebooks share the sampling and encoding code in `src/airport_encoding.py`.
