# ITSC-3162-Final-Project

# Location Encoding and Model Comparison

## Modeling plan

See the [modeling plan](MODELING_PLAN.md) for how we compare five airport encodings on the same flights using fixed model settings.

## Project Overview

This project investigates how different encoding strategies for categorical location variables affect the predictive performance of machine learning models. Location variables can contain hundreds or thousands of unique categories, making their representation an important modeling decision.

The notebook creates five datasets on the same train/test split. Each dataset contains only its encoded origin and destination features plus the arrival-delay target. No other flight columns are included as model inputs.

- **One-hot encoding:** Creates a separate binary feature for each location.
- **Ordinal encoding:** Assigns a numerical identifier to each location.
- **Frequency encoding:** Uses the airport's share of training flights as an origin or destination.
- **Target encoding:** Represents locations using statistics derived from the target variable.
- **Geographic encoding:** Represents locations using spatial information, such as coordinates or distances.

## Methodology

We will evaluate these strategies across several machine learning algorithms, including linear regression, random forests, and gradient boosting.

Each combination of encoding strategy and model will form a separate pipeline. We will use identical training and validation splits and compare predictive performance using cross-validation.

All preprocessing will be performed within the training folds to prevent data leakage.

## Research Questions

1. Which location encoding strategy produces the best predictive performance?
2. Does the optimal encoding strategy depend on the machine learning algorithm?
3. How do different encoding strategies affect computational efficiency and model complexity?

## Evaluation

For a regression task, we will use RMSE and MAE to evaluate predictive accuracy. We will also compare training times and feature dimensionality.

Our goal is to determine whether certain encoding strategies consistently outperform others or whether their effectiveness depends on the model being used.

## Expected Outcome

The project will provide an empirical comparison of location encoding methods and demonstrate how encoding choices interact with machine learning algorithms. Rather than assuming one encoding strategy is universally superior, we will identify which combinations work best for the selected prediction task.

## Data files

All datasets are stored in `data/`:

- `data/flights.csv`: flight records, including the `arr_delay` target.
- `data/airport_coordinates.csv`: airport latitude/longitude lookup for geographic encoding.

Run `Final_Project.ipynb` with the project root as the working directory so its relative data paths resolve correctly.
