"""Shared flight sampling and airport encoders for the project notebooks."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder

AIRPORT_COLUMNS = ["origin", "dest"]
RANDOM_STATE = 42
SAMPLE_SIZE = 100_000
TARGET_COLUMN = "arr_delay"


def sample_flights(flights, target_column=TARGET_COLUMN):
    """Draw the shared sample, then return its 80/20 development and test splits."""
    if target_column is not None:
        if target_column not in flights.columns or target_column in AIRPORT_COLUMNS:
            raise ValueError("Choose an existing outcome column other than origin or dest.")
        flights = flights.dropna(subset=[target_column])
    if len(flights) < SAMPLE_SIZE:
        raise ValueError(f"Need at least {SAMPLE_SIZE:,} eligible rows for sampling.")
    sample = flights.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
    return train_test_split(sample, test_size=0.20, random_state=RANDOM_STATE)


def make_encoded_datasets(train, test, coordinates, target_column=None):
    """Build airport-only datasets with the target, fitting encodings on train."""
    if not train.index.is_unique or not test.index.is_unique:
        raise ValueError("Row indices must be unique for reliable alignment.")
    if len(train.index.intersection(test.index)):
        raise ValueError("Training and test indices must be disjoint.")
    if not train.columns.equals(test.columns):
        raise ValueError("Training and test columns must match.")
    if train[AIRPORT_COLUMNS].isna().any().any() or test[AIRPORT_COLUMNS].isna().any().any():
        raise ValueError("Handle missing airport codes before encoding.")

    train_codes = train[AIRPORT_COLUMNS].astype(str)
    test_codes = test[AIRPORT_COLUMNS].astype(str)
    variants, fitted, feature_names = {}, {}, {}

    def save(name, train_features, test_features):
        # Preserve predictor names before adding the target for notebook display.
        feature_names[name] = train_features.columns.tolist()
        if target_column is not None:
            train_features = pd.concat([train[[target_column]], train_features], axis=1)
            test_features = pd.concat([test[[target_column]], test_features], axis=1)
        variants[name] = {"train": train_features, "test": test_features}

    # 1. One-hot: one indicator for each airport observed in training.
    one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.int8)
    train_matrix = one_hot.fit_transform(train_codes)
    test_matrix = one_hot.transform(test_codes)
    names = one_hot.get_feature_names_out(AIRPORT_COLUMNS)
    save("one_hot",
         pd.DataFrame.sparse.from_spmatrix(train_matrix, index=train.index, columns=names),
         pd.DataFrame.sparse.from_spmatrix(test_matrix, index=test.index, columns=names))
    fitted["one_hot"] = one_hot

    # 2. Ordinal: arbitrary category IDs, not geographic rankings.
    ordinal = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                             dtype=np.int32)
    names = [f"{col}_ordinal" for col in AIRPORT_COLUMNS]
    save("ordinal",
         pd.DataFrame(ordinal.fit_transform(train_codes), index=train.index, columns=names),
         pd.DataFrame(ordinal.transform(test_codes), index=test.index, columns=names))
    fitted["ordinal"] = ordinal

    # 3. Frequency: use only the training distribution for both splits.
    frequency_maps = {col: train_codes[col].value_counts(normalize=True)
                      for col in AIRPORT_COLUMNS}
    def frequencies(codes):
        return pd.DataFrame({f"{col}_frequency": codes[col].map(frequency_maps[col]).fillna(0.0)
                             for col in AIRPORT_COLUMNS}, index=codes.index)
    save("frequency", frequencies(train_codes), frequencies(test_codes))
    fitted["frequency"] = frequency_maps

    # 4. Smoothed target: optional until the prediction target is selected.
    if target_column is not None:
        if target_column in AIRPORT_COLUMNS:
            raise ValueError("The target must be separate from origin and dest.")
        y_train = train[target_column]
        if not pd.api.types.is_numeric_dtype(y_train) or not np.isfinite(y_train).all():
            raise ValueError("Target encoding requires finite numeric training targets.")
        if len(train) < 5:
            raise ValueError("At least five training rows are needed for cross-fitting.")
        target_encoder = TargetEncoder(target_type="continuous", smooth=20.0,
                                       cv=5, shuffle=True, random_state=RANDOM_STATE)
        names = [f"{col}_target_mean" for col in AIRPORT_COLUMNS]
        # fit_transform is intentional: fit(...).transform(...) would leak targets.
        train_encoded = target_encoder.fit_transform(train_codes, y_train)
        test_encoded = target_encoder.transform(test_codes)
        save("smoothed_target",
             pd.DataFrame(train_encoded, index=train.index, columns=names),
             pd.DataFrame(test_encoded, index=test.index, columns=names))
        fitted["smoothed_target"] = target_encoder

    # 5. Geographic: fixed latitude/longitude lookup, independent of the target.
    def geographic(codes):
        features = pd.DataFrame(index=codes.index)
        for col in AIRPORT_COLUMNS:
            for coordinate in ["latitude", "longitude"]:
                features[f"{col}_{coordinate}"] = codes[col].map(coordinates[coordinate])
        if features.isna().any().any():
            missing = sorted(set(codes.to_numpy().ravel()) - set(coordinates.index))
            raise ValueError(f"Missing airport coordinates: {missing}")
        return features
    save("geographic", geographic(train_codes), geographic(test_codes))
    fitted["geographic"] = coordinates.copy()
    return variants, fitted, feature_names
