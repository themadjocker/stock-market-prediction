from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.validation.preprocessor import TrainOnlyPreprocessor


def test_fit_and_transform_standard() -> None:
    train_df = pd.DataFrame({"f1": [10.0, 20.0, 30.0], "f2": [1.0, 2.0, 3.0]})
    val_df = pd.DataFrame({"f1": [40.0, 50.0], "f2": [4.0, 5.0]})

    preprocessor = TrainOnlyPreprocessor(scaler_type="standard")
    scaled_train = preprocessor.fit_transform(train_df)

    assert preprocessor.is_fitted is True
    # Mean of train: f1=20, f2=2. Std of train: f1=sqrt(200/3)=8.16496, f2=sqrt(2/3)=0.816496
    np.testing.assert_allclose(scaled_train.mean(), 0.0, atol=1e-7)

    scaled_val = preprocessor.transform(val_df)
    # Val values must be scaled using train mean and train std!
    train_mean_f1 = train_df["f1"].mean()
    train_std_f1 = np.std(train_df["f1"].to_numpy())
    expected_val_f1_0 = (40.0 - train_mean_f1) / train_std_f1
    np.testing.assert_allclose(scaled_val.iloc[0]["f1"], expected_val_f1_0, atol=1e-5)


def test_refitting_is_strictly_prohibited() -> None:
    train_df = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
    val_df = pd.DataFrame({"f1": [4.0, 5.0, 6.0]})

    preprocessor = TrainOnlyPreprocessor()
    preprocessor.fit(train_df)

    # Calling fit again must raise RuntimeError
    with pytest.raises(RuntimeError, match="Re-fitting is strictly prohibited"):
        preprocessor.fit(val_df)

    with pytest.raises(RuntimeError, match="Re-fitting is strictly prohibited"):
        preprocessor.fit_transform(val_df)


def test_transform_before_fit_raises() -> None:
    df = pd.DataFrame({"f1": [1.0, 2.0]})
    preprocessor = TrainOnlyPreprocessor()
    with pytest.raises(RuntimeError, match="must be fitted on training data"):
        preprocessor.transform(df)


def test_column_mismatch_raises() -> None:
    train_df = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
    bad_df = pd.DataFrame({"f1": [1.0, 2.0], "f3": [5.0, 6.0]})
    preprocessor = TrainOnlyPreprocessor()
    preprocessor.fit(train_df)

    with pytest.raises(ValueError, match="Feature columns mismatch"):
        preprocessor.transform(bad_df)
