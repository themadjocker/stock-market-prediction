from __future__ import annotations

import numpy as np
import pandas as pd

from ml.models.baseline import NaiveBaselineModel


def test_naive_baseline_zero_strategy() -> None:
    X_train = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
    y_train = pd.DataFrame({"target_return_1d": [0.01, -0.02, 0.05]})
    X_val = pd.DataFrame({"f1": [4.0, 5.0]})

    model = NaiveBaselineModel(strategy="zero")
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    assert len(preds) == 2
    np.testing.assert_array_equal(preds, [0.0, 0.0])


def test_naive_baseline_mean_strategy() -> None:
    X_train = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
    y_train = pd.DataFrame({"target_return_1d": [0.02, 0.04, 0.06]})
    X_val = pd.DataFrame({"f1": [4.0, 5.0]})

    model = NaiveBaselineModel(strategy="mean")
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    np.testing.assert_allclose(preds, [0.04, 0.04])


def test_naive_baseline_majority_strategy() -> None:
    X_train = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
    y_train = pd.DataFrame({"target_up_1d": [1.0, 1.0, 0.0]})
    X_val = pd.DataFrame({"f1": [4.0]})

    model = NaiveBaselineModel(strategy="majority")
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    assert preds[0] == 1.0
