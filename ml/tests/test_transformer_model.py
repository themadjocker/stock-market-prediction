from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.models.transformer_model import TransformerModel


def make_data(n_samples: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "f1": np.sin(np.linspace(0, 10, n_samples)),
            "f2": np.cos(np.linspace(0, 10, n_samples)),
        }
    )
    y = pd.DataFrame({"target_return_1d": 0.1 * X["f1"] + 0.05 * X["f2"]})
    return X, y


def test_transformer_model_fit_predict() -> None:
    X, y = make_data(40)
    X_train, y_train = X.iloc[:30], y.iloc[:30]
    X_val, y_val = X.iloc[30:], y.iloc[30:]

    model = TransformerModel(
        lookback=5,
        d_model=16,
        nhead=2,
        dim_feedforward=32,
        epochs=3,
        batch_size=8,
        seed=42,
    )
    model.fit(X_train, y_train, X_val, y_val)

    preds = model.predict(X_val)
    assert len(preds) == len(X_val)
    assert np.isfinite(preds).all()


def test_transformer_model_deterministic() -> None:
    X, y = make_data(35)
    model1 = TransformerModel(
        lookback=4,
        d_model=16,
        nhead=2,
        epochs=3,
        batch_size=8,
        seed=42,
    )
    model1.fit(X.iloc[:25], y.iloc[:25])
    preds1 = model1.predict(X.iloc[25:])

    model2 = TransformerModel(
        lookback=4,
        d_model=16,
        nhead=2,
        epochs=3,
        batch_size=8,
        seed=42,
    )
    model2.fit(X.iloc[:25], y.iloc[:25])
    preds2 = model2.predict(X.iloc[25:])

    np.testing.assert_allclose(preds1, preds2, atol=1e-5)


def make_panel_data(
    n_days: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2026-01-01", periods=n_days, freq="B")
    rows = []
    for ticker, offset in [("AAA", 0.0), ("BBB", 100.0)]:
        for i, date in enumerate(dates):
            rows.append({"f1": offset + float(i), "ticker": ticker, "session_date": date})
    frame = pd.DataFrame(rows)
    X = frame[["f1"]].copy()
    metadata = frame[["ticker", "session_date"]].copy()
    y = pd.DataFrame({"target_return_1d": X["f1"] * 0.001})
    return X, y, metadata


def test_transformer_model_uses_metadata_and_train_history_for_validation() -> None:
    X, y, metadata = make_panel_data(10)
    train_n = 14
    model = TransformerModel(lookback=3, d_model=16, nhead=2, epochs=2, batch_size=8, seed=42)
    model.fit(
        X.iloc[:train_n].reset_index(drop=True),
        y.iloc[:train_n].reset_index(drop=True),
        X.iloc[train_n:].reset_index(drop=True),
        y.iloc[train_n:].reset_index(drop=True),
        metadata_train=metadata.iloc[:train_n].reset_index(drop=True),
        metadata_val=metadata.iloc[train_n:].reset_index(drop=True),
    )
    preds = model.predict(
        X.iloc[train_n:].reset_index(drop=True),
        metadata=metadata.iloc[train_n:].reset_index(drop=True),
        history_X=X.iloc[:train_n].reset_index(drop=True),
        history_metadata=metadata.iloc[:train_n].reset_index(drop=True),
    )
    assert len(preds) == len(X) - train_n
    assert np.isfinite(preds).all()


def test_transformer_model_rejects_panel_prediction_without_metadata() -> None:
    X, y, metadata = make_panel_data(8)
    model = TransformerModel(lookback=3, d_model=16, nhead=2, epochs=1, batch_size=8, seed=42)
    model.fit(X, y, metadata_train=metadata)
    with pytest.raises(ValueError):
        model.predict(X, metadata=None)


def test_transformer_passes_metadata_and_validation_endpoints_to_sequence_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X, y, metadata = make_panel_data(8)
    calls: list[dict[str, object]] = []
    import ml.models.transformer_model as module

    original = module.construct_lookback_sequences

    def spy(*args, **kwargs):
        calls.append(
            {
                "metadata": kwargs.get("metadata"),
                "endpoint_indices": kwargs.get("endpoint_indices"),
            }
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "construct_lookback_sequences", spy)
    model = TransformerModel(lookback=3, d_model=16, nhead=2, epochs=1, batch_size=8, seed=42)
    model.fit(
        X.iloc[:12].reset_index(drop=True),
        y.iloc[:12].reset_index(drop=True),
        X.iloc[12:].reset_index(drop=True),
        y.iloc[12:].reset_index(drop=True),
        metadata_train=metadata.iloc[:12].reset_index(drop=True),
        metadata_val=metadata.iloc[12:].reset_index(drop=True),
    )
    assert any(call["metadata"] is not None for call in calls)
    assert any(call["endpoint_indices"] is not None for call in calls)


def test_transformer_supports_binary_classification() -> None:
    X = pd.DataFrame({"f1": np.linspace(-1, 1, 40), "f2": np.cos(np.linspace(0, 4, 40))})
    y = pd.DataFrame({"target_up_1d": (X["f1"] > 0).astype(float)})
    model = TransformerModel(lookback=4, d_model=16, nhead=2, epochs=2, batch_size=8, seed=42)
    model.fit(X.iloc[:30], y.iloc[:30], X.iloc[30:], y.iloc[30:])
    preds = model.predict(X.iloc[30:])
    assert len(preds) == 10
    assert np.isfinite(preds).all()
    assert np.all((preds >= 0.0) & (preds <= 1.0))
